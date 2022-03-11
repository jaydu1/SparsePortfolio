import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
import time
import sys
import os

sys.path.append("./")
path_data = './data/russell/'
path_result = 'result/'
os.makedirs(path_result, exist_ok=True)

k = int(sys.argv[1])
method_list = ['MV-P', 'MV-LS', 'MV-NLS']
method = method_list[k]

n_folds = 10
n_lam_MV = 100
lam_MVs = np.logspace(np.log10(1e-3), np.log10(1e2), num=n_lam_MV)[::-1]

df_close = pd.read_csv(path_data+'russell2000_PRC.csv', index_col=[0])
df_open = pd.read_csv(path_data+'russell2000_OPENPRC.csv', index_col=[0])
df = df_close/df_open
T, d = df.shape

id_begin = np.where(df.index>=20050101)[0][0]
df_hold = pd.read_csv(path_data+'russell2000_RET.csv', index_col=[0]) + 1
df_listed = pd.read_csv(path_data+'russell2000_listed.csv', index_col=[0])

id_recal = []
id_train_begin = []
id_codes_list = []
id_test_end = []
for year in range(2005,2021):
    for month in [1,7]:
        date = int('%d%02d01'%(year,month))        
        date_train = int('%d%02d01'%(year-1,month))
        codes = df_listed.columns[
            (np.all(df_listed[
                (df_listed.index>=int('%d%02d01'%(year-int(month==1),(month+6)%12)))&
                (df_listed.index<=date)]==1, axis=0))]
        _df = df[codes].iloc[(df.index>=date_train)&(df.index<date)].copy()
        codes = codes[
            ~np.any(np.isnan(_df), axis=0)
        ]
        id_codes_list.append(
            np.array([np.where(np.array(list(df.columns))==i)[0][0] for i in codes])
            )
        id_recal.append(np.where(df.index>=date)[0][0])
        id_train_begin.append(np.where(df.index>=date_train)[0][0])
        id_test_end.append(np.where(df.index<int('%d%02d01'%(year+int(month==7),(month+6)%12)))[0][-1])
        

n_recal = len(id_recal)

df = df.fillna(1.)
df_hold.iloc[id_recal,:] = df.iloc[id_recal,:].copy()
df_hold = df_hold.fillna(1.)
test_date = np.array(df.index[id_begin:id_test_end[-1]+1])


import numpy as np
import pandas as pd
import nonlinshrink as nls #nonlinear shrinkage
from sklearn.covariance import LedoitWolf #linear shrinkage
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
from cvxopt import matrix
from cvxopt import solvers
import time


def proj(w):
    d = w.shape[0]
    sort_w = -np.sort(-w, axis=None)
    tmp = (np.cumsum(sort_w) - 1) * (1.0/np.arange(1,d+1))
    rho = np.sum(sort_w > tmp) - 1
    w = np.maximum(w - tmp[rho], 0)
    return w


def mean_var_opt(Sigma, mu, lam_MV=0.0, maxiters=int(1e3)):
    d = Sigma.shape[0]

    eps = 1e-6
    delta = np.finfo(np.float64).eps
    while True:
        try:
            P = matrix(lam_MV * Sigma, tc='d')
            q = matrix(-mu, tc='d')
            G = matrix(-np.eye(d), tc='d')
            h = matrix(np.zeros(d), tc='d')
            A = matrix(np.ones(d).reshape((1,-1)), tc='d')
            b = matrix(np.ones(1), tc='d')
            sol = solvers.qp(P,q,G,h,A,b, options={'show_progress': False, 
                                      'abstol':1e-12, 'reltol':1e-11, 
                                      'maxiters':maxiters, 'feastol':1e-16})
            w = np.array(sol['x']).flatten()
            w[w<=delta] = 0.
            w = proj(w)
            break
        except:
            print('singular')
            Sigma = Sigma + np.identity(d) * eps
            eps *= 10
    
    return w


def cov(X, method):
    '''
    Parameters
    ----------
    X : np.array
        The sample matrix with size \(n, p\).
    method : str
        The method used to estimate the covariance.

    Returns
    ----------
    Cov : np.array
        The estimated covariance matrix.
    '''
    if method.startswith('MV-P'):
        return np.cov(X, rowvar = False)
    elif method.startswith('MV-LS'):
        cov = LedoitWolf(assume_centered = False).fit(X) 
        return cov.covariance_
    elif method.startswith('MV-NLS'):
        return nls.shrink_cov(X)


def eval(X_train, X_val, method, lam_MV=0.0, maxiters=int(1e3)):
    Sigma = cov(X_train, method)
    mu = np.mean(X_train, axis=0)-1
    w = mean_var_opt(Sigma, mu, lam_MV, maxiters)
    return np.dot(X_val-1, w), w


def fit(i, df, df_hold, method):
    ws_test = np.empty((1, d))
    ws_test.fill(np.nan)

    X = df.iloc[id_train_begin[i]:id_recal[i],:].values[:,id_codes_list[i]].copy()
    X_test = df_hold.iloc[id_recal[i]:id_test_end[i]+1,:].values[:,id_codes_list[i]].copy()
    
    t1 = time.time()
    kf = TimeSeriesSplit(n_splits=n_folds, test_size=None)
    
    score_val = np.zeros((n_folds, n_lam_MV))
    for j, (train_index, val_index) in tqdm(enumerate(kf.split(X)), total = n_folds):
        X_train, X_val = X[train_index], X[val_index]

        for k, lam_MV in enumerate(lam_MVs):
            tmp, w = eval(X_train, X_val, method, lam_MV, int(1e2))

            # by returns
            # score_val[j,k] = -np.inf if np.sum(w)==0. else np.mean(tmp)

            # by sharpe ratio                    
            ret = np.log(tmp + 1)
            score_val[j,k] = 0. if np.sum(w)==0 else np.mean(ret)/np.std(ret)

    id_lam = np.argmax(np.median(score_val, axis=0))
    lam_MV = np.array([lam_MVs[id_lam]])
    score_test, ws_test[0, id_codes_list[i]] = eval(X, X_test, method, lam_MV)
    time_elapsed = time.time() - t1

    ret = np.log(score_test+1)
    print(np.nancumprod(score_test+1)[-1], np.mean(ret)/np.std(ret))
    return score_test, ws_test, time_elapsed


from joblib import Parallel, delayed
with Parallel(n_jobs=-1, verbose=100) as parallel:
    # for i in range(3):
    #     method = method_list[i]
    print(method)

    out = parallel(delayed(fit)(i, df, df_hold, method) for i in range(len(id_recal)))        
    score_test_list, ws_test_list, time_elapsed = zip(*out)
    score_test_list = np.concatenate(score_test_list, axis=0)
    ws_test_list = np.concatenate(ws_test_list, axis=0)
    time_elapsed = np.array(time_elapsed)

    np.savez(path_result+'res_russell_%s.npz'%(method), 
            score_test_list=score_test_list, ws_test_list=ws_test_list, test_date=test_date, times=time_elapsed)
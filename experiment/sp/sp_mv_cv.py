import numpy as np
import pandas as pd
import nonlinshrink as nls #nonlinear shrinkage
from sklearn.covariance import LedoitWolf #linear shrinkage
from sklearn.model_selection import TimeSeriesSplit

from cvxopt import matrix
from cvxopt import solvers
import sys

def proj(w):
    d = w.shape[0]
    sort_w = -np.sort(-w, axis=None)
    tmp = (np.cumsum(sort_w) - 1) * (1.0/np.arange(1,d+1))
    rho = np.sum(sort_w > tmp) - 1
    w = np.maximum(w - tmp[rho], 0)
    return w


def mean_var_opt(Sigma, mu, lam_MV=0.0):
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
                                      'maxiters':int(1e4), 'feastol':1e-16})
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


def eval(X_train, X_val, method, lam_MV=0.0):
    Sigma = cov(X_train, method)
    mu = np.mean(X_train, axis=0)-1
    w = mean_var_opt(Sigma, mu, lam_MV)
    return np.dot(X_val-1, w), w


def fit(i, df, df_hold, method):
    i_len = np.minimum((i + 1) * n_days_hold, len(test_date)) - i * n_days_hold
    idx = id_recal[i]

    score_test = np.zeros((i_len, ))
    ws_test = np.empty((1, d))
    ws_test.fill(np.nan)

    X = df.iloc[idx-n_days_train:idx,:].values[:,id_codes_list[i]].copy()
    X_test = df_hold.iloc[idx:idx+n_days_hold,:].values[:,id_codes_list[i]].copy()
    
    kf = TimeSeriesSplit(n_splits=n_folds, test_size=None)
    
    score_val = np.zeros((n_folds, n_lam_MV))
    for j, (train_index, val_index) in enumerate(kf.split(X)):
        X_train, X_val = X[train_index], X[val_index]

        for k, lam_MV in enumerate(lam_MVs):
            tmp, w = eval(X_train, X_val, method, lam_MV)

            # by returns
            # score_val[j,k] = -np.inf if np.sum(w)==0. else np.mean(tmp)

            # by sharpe ratio                    
            
            ret = np.log(tmp + 1)
            score_val[j,k] = 0. if np.sum(w)==0 else np.mean(ret)/np.std(ret)

    id_lam = np.argmax(np.median(score_val, axis=0))
    lam_MV = np.array([lam_MVs[id_lam]])
    score_test[:], ws_test[0, id_codes_list[i]] = eval(X, X_test, method, lam_MV)
                
    return score_test, ws_test

sys.path.append("./")
path_data = 'data/sp/'

k = int(sys.argv[1])
method_list = ['MV-P', 'MV-LS', 'MV-NLS']
method = method_list[k]

n_days_train = 120
n_days_hold = 63

df_close = pd.read_csv(path_data+'sp500_PRC.csv', index_col=[0])
df_open = pd.read_csv(path_data+'sp500_OPENPRC.csv', index_col=[0])
df = df_close/df_open
T, d = df.shape

id_begin = np.where(df.index>=20110101)[0][0]
df_hold = pd.read_csv(path_data+'sp500_RET.csv', index_col=[0]) + 1
id_recal = np.arange(id_begin, len(df.index), n_days_hold)
df_hold.iloc[id_recal,:] = df.iloc[id_recal,:].copy()
df_hold = df_hold.fillna(1.)
n_recal = len(id_recal)

test_date = np.array(df.index[id_begin:])

df_listed = pd.read_csv(path_data+'sp500_listed.csv', index_col=[0])
df_listed.index = np.array(pd.to_datetime(df_listed.index).strftime('%Y%m%d')).astype(int)
id_codes_list = []
for idx in id_recal:
    codes = df_listed.columns[
        (np.all(df_listed[
            (df_listed.index>=df.index[idx-n_days_hold*4])&
            (df_listed.index<=df.index[idx])]==1, axis=0))]
    codes = codes[~df.iloc[idx-n_days_train:idx,:].loc[:,codes].isnull().any()]
    id_codes_list.append(
        np.array([np.where(np.array(list(df.columns))==i)[0][0] for i in codes])
        )

method_list = ['MV-P', 'MV-LS', 'MV-NLS']

n_folds = 5
n_lam_MV = 100
lam_MVs = np.logspace(np.log10(1e-3), np.log10(1e2), num=n_lam_MV)[::-1]

from joblib import Parallel, delayed
with Parallel(n_jobs=-1, verbose=100) as parallel:
    # for i in range(3):
    #     method = method_list[i]
    print(method)

    out = parallel(delayed(fit)(i, df, df_hold, method) for i in range(len(id_recal)))        
    score_test_list, ws_test_list = zip(*out)
    score_test_list = np.concatenate(score_test_list, axis=0)
    ws_test_list = np.concatenate(ws_test_list, axis=0)

    np.savez('result/res_sp_%s.npz'%(method), 
            score_test_list=score_test_list, ws_test_list=ws_test_list, test_date=test_date)
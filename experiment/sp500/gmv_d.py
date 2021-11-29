import numpy as np
import pandas as pd
import nonlinshrink as nls #nonlinear shrinkage
from sklearn.covariance import LedoitWolf #linear shrinkage

from cvxopt import matrix
from cvxopt import solvers


def proj(w):
    d = w.shape[0]
    sort_w = -np.sort(-w, axis=None)
    tmp = (np.cumsum(sort_w) - 1) * (1.0/np.arange(1,d+1))
    rho = np.sum(sort_w > tmp) - 1
    w = np.maximum(w - tmp[rho], 0)
    return w


def mean_var_opt(Sigma):
    d = Sigma.shape[0]

    eps = 1e-6
    delta = np.finfo(np.float64).eps
    while True:
        try:
            P = matrix(Sigma, tc='d')
            q = matrix(np.zeros(d), tc='d')
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
    if method.startswith('GMV-P'):
        return np.cov(X, rowvar = False)
    elif method.startswith('GMV-LS'):
        cov = LedoitWolf(assume_centered = False).fit(X) 
        return cov.covariance_
    elif method.startswith('GMV-NLS'):
        return nls.shrink_cov(X)


def eval(X_train, X_val, method):
    Sigma = cov(X_train, method)
    w = mean_var_opt(Sigma)
    return np.dot(X_val, w) - 1, w


def fit(i, df, df_hold, method):
    i_len = np.minimum((i + 1) * n_days_hold, len(test_date)) - i * n_days_hold
    idx = id_recal[i]

    score_test = np.zeros((i_len, ))
    ws_test = np.zeros((1, d))

    X = df.iloc[idx-n_days_train:idx,:].values.copy()
    X_test = df_hold.iloc[idx:idx+n_days_hold,:].values.copy()
    
    score_test[:], ws_test[0, :] = eval(X, X_test, method)
                
    return score_test, ws_test


n_days_train = 120
n_days_hold = 63

df_close = pd.read_csv('data/sp500/PRC.csv', index_col=[0])
df_open = pd.read_csv('data/sp500/OPENPRC.csv', index_col=[0])
df = df_close/df_open

T, d = df.shape

id_begin = np.where(df.index>=20110101)[0][0]
df_hold = pd.read_csv('data/sp500/RET.csv', index_col=[0]) + 1
id_recal = np.arange(id_begin, len(df.index), n_days_hold)
df_hold.iloc[id_recal,:] = df.iloc[id_recal,:].copy()
n_recal = len(id_recal)

test_date = np.array(df.index[id_begin:])
method_list = ['GMV-P', 'GMV-LS', 'GMV-NLS']

n_lam_MV = 100
lam_MVs = np.logspace(np.log10(1e-3), np.log10(1e2), num=n_lam_MV)[::-1]

from joblib import Parallel, delayed
with Parallel(n_jobs=-1, verbose=100) as parallel:
    for i in range(3):
        method = method_list[i]
        print(method)

        out = parallel(delayed(fit)(i, df, df_hold, method) for i in range(len(id_recal)))        
        score_test_list, ws_test_list = zip(*out)
        score_test_list = np.concatenate(score_test_list, axis=0)
        ws_test_list = np.concatenate(ws_test_list, axis=0)

        np.savez('result/res_%s_%d_%d.npz'%(method,n_days_train,n_days_hold), 
                score_test_list=score_test_list, ws_test_list=ws_test_list, test_date=test_date)
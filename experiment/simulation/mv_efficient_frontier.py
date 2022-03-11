import pandas as pd
import numpy as np
from spo import spo_l1_path
import sys
import os
from tqdm import tqdm
import sys
sys.path.append("./")

df = pd.read_csv('data/nyse/NYSE.csv', index_col=[0])
X = df.values
X = X / np.max(np.abs(X))
X = X + 1


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


n, d = X.shape
Sigma = cov(X, 'MV-P')
mu = np.mean(X, axis=0)-1

n_lambdas = 100
lambdas = [ 10**(5.0*t/n_lambdas-1.0) for t in range(n_lambdas) ]
ws_list = []
times = []
for lam in tqdm(lambdas):
    t1 = time.time()
    ws = mean_var_opt(Sigma, mu, lam) 
    times.append(time.time() - t1)
    ws_list.append(ws)
ws_list = np.array(ws_list)


np.savez('result/res_mv_efficient_frontier.npz',                         
            ws_list=ws_list,
            lambdas=lambdas,
            times = times
        )
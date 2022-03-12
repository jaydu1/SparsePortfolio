from typing import Optional

import pandas as pd
import numpy as np
from numpy import linalg as LA
import numba as nb
from tqdm import tqdm
import time 
import sys
import os
sys.path.append("./")
path_result = 'result/'
os.makedirs(path_result, exist_ok=True)

from spo.utils import np_type_f, np_type_i, nb_type_f, nb_type_i, FUNC_LOG, FUNC_EXP
from spo.objective import Objective
from spo.utils import build_lambdas
from spo.spo import dual_scaling, dual_gap, screening, prox_gd

@nb.jit(nb.types.Tuple((nb_type_f, nb_type_f[::1], nb_type_f))(
    Objective.class_type.instance_type, nb_type_f[:,::1], nb_type_f[::1], nb_type_f[::1],
    nb_type_f[::1], nb_type_f, 
    nb_type_f, nb_type_i, nb.boolean), nopython=True, fastmath=True)
def track_one_path(obj, X, w, theta0, 
             norm2_X, L_dh,        
             lam_1, max_iter, screen):
    """
        We minimize
        f(w) + lam_11 ||w||_1
        where f(w) = - sum_{i=1}^n log(X_i w + eta) and
    """
    n_samples, n_features = X.shape
    n_active_features = np.ones(max_iter+1, dtype=nb_type_f) * n_features
    
    gap = np.inf
    
    w_old = w.copy()
    w_old_old = w_old.copy()
    theta = theta0.copy()
    XTtheta = np.dot(X.T, theta)
    dual_scale = dual_scaling(XTtheta, lam_1)
    alpha = np.min(obj.comp_dual_neg_hess(theta0/dual_scale, lam_1))

    disabled_features = np.zeros(n_features, dtype=nb_type_i)
    
    for n_iter in range(max_iter):
        id_features = (disabled_features == 0)
        screened_w = w[id_features]
        screened_X = X[:, id_features]

        # Update dual variables
        theta[:] = obj.comp_dual_var(screened_X, screened_w)
        XTtheta[:] = np.dot(X.T, theta)

        dual_scale = dual_scaling(XTtheta, lam_1)
        
        _, _, gap = dual_gap(obj, screened_X, screened_w, theta, dual_scale, 
                            disabled_features, lam_1)
            
        if screen:
            r = np.sqrt(2 * gap / alpha)
            XTcenter = XTtheta / dual_scale
            
            w, n_active_features[n_iter+1] = screening(w, XTcenter, r,
                    norm2_X, n_active_features[n_iter], disabled_features)
            
        if gap==0.:
            n_active_features[n_iter+1:] = n_active_features[n_iter+1]
            break
            
        # The local Lipschitz constant of h's gradient.
        L_dh = LA.norm(X * np.sqrt(theta).reshape((-1,1)), ord=2) ** 2

        w, _ = prox_gd(w, -XTtheta, L_dh,
            disabled_features, lam_1, w_old, w_old_old)

    w = obj.proj(w)

    return gap, n_active_features[1:], n_iter


def spo_l1_path(X, func: int = 0, a: float = 1.0, 
    lambdas: Optional[list] = None, n_lambdas: int = 100, delta: float = 2.0,
    max_iter: int = int(1e5), tol: float = 1e-4, screen: bool = True):
    """Sparse portfolio optimization with proximal gradient descent

    The formulation reads:

    f(w) + lambda_1 norm(w, 1)
    where f(w) = - 1/n * sum_{j=1}^n log(w^TX_j).

    Parameters
    ----------
    X : {array-like}, shape (n_samples, n_features)
        Training data of growth rates. 

    func : int, optional
        The utility function: 0 - log, 1 - exp.

    a : float, optional
        The risk aversion parameter for exp utility.

    lambdas : ndarray, optional, shape (n_lambdas,)
        List of lambdas where to compute the models.

    n_lambdas : int, optional
        The number of lambdas.

    delta : float, optional
        The log spacing of lambdas.

    max_iter : int, optional
        The maximum number of iterations.

    tol : float, optional
        Prescribed accuracy on the duality gap.

    screen : boolean, optional
        Whether use screening rules or not.

    f : int, optional
        The screening rule will be execute at each f pass on the data


    Returns
    -------
    ws : array, shape (n_features, n_lambdas)
        Coefficients along the path.

    lambdas : ndarray, shape (n_lambdas,)
        The list of lambdas.

    gaps : array, shape (n_lambdas,)
        The dual gaps at the end of the optimization for each lambda.

    n_iters : array-like, shape (n_lambdas,)
        The number of iterations taken by the block coordinate descent
        optimizer to reach the specified accuracy for each lambda.

    n_active_features : array, shape (n_lambdas,)
        Number of active variables.

    """
    n_samples, n_features = X.shape
    # Use C-contiguous data to avoid unnecessary memory duplication.
    X = np.ascontiguousarray(X)
    minX = np.min(X)    
    if minX<=0.:
        raise ValueError('The growth rate X must be positive.') 
    X = X - minX
    
    obj = Objective(n_samples, minX, func, a)
    w_init = np.zeros(n_features, dtype=np_type_f)
    theta0 = obj.comp_dual_var(X, w_init)
    theta = theta0.copy()
    XTtheta = np.dot(X.T, theta)
    
    if lambdas is None:
        lambdas = build_lambdas(XTtheta, n_lambdas, delta)

    n_lambdas = np_type_i(lambdas.shape[0])

    # Useful precomputation
    norm2_X = np.sum(X**2, axis=0)
    L_dh = LA.norm(X * np.sqrt(theta).reshape((-1,1)), ord=2) ** 2
    
    ws = np.zeros((n_features, n_lambdas), dtype=np_type_f)
    gaps = np.ones(n_lambdas, dtype=np_type_f)
    n_active_features = np.zeros((n_lambdas, max_iter), dtype=np_type_i)
    n_iters = np.zeros(n_lambdas, dtype=np_type_i)
    times = np.zeros(n_lambdas, dtype=np_type_f)
    times.fill(np.nan)
    for t in tqdm(range(n_lambdas)):
        t1 = time.time()
        gaps[t], n_active_features[t,:], \
        n_iters[t] = track_one_path(obj, X, w_init, theta, 
                norm2_X, L_dh,      
                lambdas[t], max_iter, screen)
        times[t] = time.time() -  t1
        ws[:, t] = w_init.copy()
        theta = theta0.copy()
        w_init[:] = 0.        
    return ws, lambdas, gaps, n_iters, n_active_features, times


df = pd.read_csv('data/nyse/NYSE.csv', index_col=[0])
X = df.values
X = X/np.max(np.abs(X))
X += 1

n_lambdas = 100
func_names = ['LOG', 'EXP']
for func in range(2):
    if func == 0:
        a_list = [1.]
    else:
        a_list = [5e-2,1e-1,5e-1,1.,1.5]
    for a in a_list:
        ws, lambdas, gaps, n_iters, n_actives, times = spo_l1_path(X, func, a, None, n_lambdas, screen=True, max_iter=int(1e5))
        np.savez('result/res_ex1_%s_%.2f.npz'%(func_names[func],a), 
                    ws=ws, lambdas=lambdas, gaps=gaps, n_iters=n_iters, n_actives=n_actives, times=times
                )
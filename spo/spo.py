from typing import Optional

import numpy as np
from numpy import linalg as LA
import numba as nb
from tqdm import tqdm

from .objective import Objective
from .utils import soft_thresholding, build_lambdas
from .utils import np_type_f, np_type_i, nb_type_f, nb_type_i, FUNC_LOG, FUNC_EXP


@nb.jit(nb_type_f(nb_type_f[::1], nb_type_f), nopython=True, fastmath=True)
def dual_scaling(XTtheta, lam_1):
    dual_scale = np.maximum(lam_1, LA.norm(XTtheta, np.inf))
    return dual_scale


@nb.jit(nb.types.UniTuple(nb_type_f,3)(
    Objective.class_type.instance_type, nb_type_f[:,::1], nb_type_f[::1], nb_type_f[::1], 
    nb_type_f, nb_type_f), nopython=True, fastmath=True)
def dual_gap(obj, screened_X, screened_w, theta, dual_scale, lam_1):
    pval = obj.val(screened_X, screened_w)
    pval += lam_1 * LA.norm(screened_w, ord=1)
    dval = obj.conj(lam_1 * theta / dual_scale)     
    gap = np.maximum(pval - dval, nb_type_f(0.))

    return pval, dval, gap


@nb.jit(nb.types.Tuple((nb_type_f[::1], nb_type_i))(
    nb_type_f[::1], nb_type_f[::1], nb_type_f,
    nb_type_f[::1], nb_type_i, nb_type_i[::1]), nopython=True, fastmath=True)
def screening(w, XTcenter, r,
              norm2_X, n_active_features, disabled_features):
    '''
    
    Return:
        w
        n_active_groups
        n_active_features
    Modified inplace:
        XTtheta
        disabled_groups
        disabled_features        
    '''
    n_features = w.shape[0]

    # Safe rule for Feature level
    for j in range(n_features):

        if disabled_features[j] == 1:
            continue

        r_normX_j = r * np.sqrt(norm2_X[j])        
        if r_normX_j > 1.:
            continue

        if np.maximum(XTcenter[j], 0) + r_normX_j < 1.:
            w[j] = 0.
            disabled_features[j] = 1
            n_active_features -= 1
    return w, n_active_features    


@nb.jit(nb.types.Tuple((nb_type_f[::1], nb.boolean))(
    nb_type_f[::1], nb_type_f[::1], nb_type_f,
    nb_type_i[::1], nb_type_f, nb_type_f[::1], nb_type_f[::1]), nopython=True, fastmath=True)
def prox_gd(w, grad, L_dh,
            disabled_features, lam_1, w_old, w_old_old):  
    n_features = w.shape[0]
    is_diff = False

    # coordinate wise soft tresholding with nonnegative constraints
    thres = lam_1 / L_dh
    for j in range(n_features):
        if disabled_features[j] == 1:
            continue
        
        w_old_old[j] = w_old[j]
        w_old[j] = w[j]

        # ADMM for PGD
        # tmp = w[j] - grad[j] / L_dh
        # if tmp - thres > 0.:
        #     tmp = tmp - thres
        #     pass
        # else:
        #     xp = 1e-8
        #     z = tmp
        #     u = 0.
        #     iter = 0
        #     e_rel = 1.
        #     while iter<1e4 and e_rel>1e-4:
        #         iter += 1
        #         x = np.sign(z - u) * np.maximum(np.abs(z - u) - thres, nb_type_f(0.))                
        #         z = np.maximum(tmp + x + u, 0.)/2.0
        #         u = u + x - z  
        #         e_rel = np.abs((x - xp)/(xp+1e-8))
        #         xp = x
        #     tmp = x                
        # w[j] = tmp 
        tmp = np.maximum(w[j] - (grad[j] / L_dh + thres), 0)
        if (w[j]-w_old_old[j])*(tmp-w[j])<0:
            if tmp-w[j]>0:
                w[j] = (w[j] + np.minimum(w_old_old[j], tmp)) / nb_type_f(2.)
            else:
                w[j] = (w[j] + np.maximum(w_old_old[j], tmp)) / nb_type_f(2.)
        elif tmp>w[j]>w_old_old[j]:
            w[j] = 2 * tmp - w[j]
        else:
            w[j] = tmp
        
        if is_diff is False and w[j]!=w_old[j]:
            is_diff = True        

    return w, is_diff  


@nb.jit(nb.types.UniTuple(nb_type_f,3)(
    Objective.class_type.instance_type, nb_type_f[:,::1], nb_type_f[::1], nb_type_f[::1],
    nb_type_f[::1], nb_type_f, 
    nb_type_f, nb_type_i, nb_type_f, nb_type_f, nb.boolean), nopython=True, fastmath=True)
def track_one_path(obj, X, w, theta0, 
             norm2_X, L_dh,        
             lam_1, max_iter, f, tol, screen):
    """
        We minimize
        f(w) + lam_11 ||w||_1
        where f(w) = - sum_{i=1}^n log(X_i w + eta) and
    """
    _, n_features = X.shape
    n_active_features = n_features
    
    final_pass = False
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

        if f != 0 and (n_iter % f == 0 or final_pass):   
            dual_scale = dual_scaling(XTtheta, lam_1)
            
            _, _, gap = dual_gap(obj, screened_X, screened_w, theta, dual_scale, lam_1)

            if gap <= tol or final_pass:
                final_pass = True
                break

            if screen:
                r = np.sqrt(2 * gap / alpha)
                XTcenter = XTtheta / dual_scale
                
                w, n_active_features = screening(w, XTcenter, r,
                        norm2_X, n_active_features, disabled_features)
            
            # The local Lipschitz constant of h's gradient.
            L_dh = LA.norm(X * np.sqrt(theta).reshape((-1,1)), ord=2) ** 2

        if final_pass:
            break

        w, is_diff = prox_gd(w, -XTtheta, L_dh,
            disabled_features, lam_1, w_old, w_old_old)
        
        if not is_diff:
            final_pass = True

    w = obj.proj(w)

    return gap, n_active_features, n_iter


def spo_l1_path(X, func: int = 0, a: float = 1.0, 
    lambdas: Optional[list] = None, n_lambdas: int = 100, delta: float = 2.0,
    max_iter: int = int(1e5), tol: float = 1e-4, screen: bool = True, f: int = 30, verbose: bool = True):
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
    minX = np.min(X)    
    if minX<=0.:
        raise ValueError('The growth rate X must be positive.') 
    X = X - minX
    # Use C-contiguous data to avoid unnecessary memory duplication.
    X = np.ascontiguousarray(X)

    obj = Objective(n_samples, minX, func, a)
    w_init = np.zeros(n_features, dtype=np_type_f)
    theta = obj.comp_dual_var(X, w_init)
    XTtheta = np.dot(X.T, theta)
    
    if lambdas is None:
        lambdas = build_lambdas(XTtheta, n_lambdas, delta)

    n_lambdas = np_type_i(lambdas.shape[0])

    # Useful precomputation
    norm2_X = np.sum(X**2, axis=0)
    L_dh = LA.norm(X * np.sqrt(theta).reshape((-1,1)), ord=2) ** 2
    
    ws = np.zeros((n_features, n_lambdas), dtype=np_type_f)
    gaps = np.ones(n_lambdas, dtype=np_type_f)
    n_active_features = np.zeros(n_lambdas, dtype=np_type_i)
    n_iters = np.zeros(n_lambdas, dtype=np_type_i)

    for t in tqdm(range(n_lambdas), disable=not verbose):
        gaps[t], n_active_features[t], \
        n_iters[t] = track_one_path(obj, X, w_init, theta, 
                norm2_X, L_dh,      
                lambdas[t], max_iter, f, tol, screen)

        ws[:, t] = w_init.copy()

    return ws, lambdas, gaps, n_iters, n_active_features


def spo_nw_min(X, func: int = 0, a: float = 1.0, 
    lambdas: Optional[list] = None, n_lambdas: int = 100, delta: float = 2.0,
    max_iter: int = int(1e5), tol: float = 1e-4, screen: bool = True, f: int = 30, verbose: bool = True,
    nw_min=5):
    """Modified version with nw_min requirement.

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
    minX = np.min(X)    
    if minX<=0.:
        raise ValueError('The growth rate X must be positive.') 
    X = X - minX
    # Use C-contiguous data to avoid unnecessary memory duplication.
    X = np.ascontiguousarray(X)

    obj = Objective(n_samples, minX, func, a)
    w_init = np.zeros(n_features, dtype=np_type_f)
    theta = obj.comp_dual_var(X, w_init)
    XTtheta = np.dot(X.T, theta)
    
    if lambdas is None:
        lambdas = build_lambdas(XTtheta, n_lambdas, delta)

    n_lambdas = np_type_i(lambdas.shape[0])

    # Useful precomputation
    norm2_X = np.sum(X**2, axis=0)
    L_dh = LA.norm(X * np.sqrt(theta).reshape((-1,1)), ord=2) ** 2
    
    ws = np.zeros((n_features, n_lambdas), dtype=np_type_f)
    gaps = np.ones(n_lambdas, dtype=np_type_f)
    n_active_features = np.zeros(n_lambdas, dtype=np_type_i)
    n_iters = np.zeros(n_lambdas, dtype=np_type_i)

    for t in tqdm(range(n_lambdas), disable=not verbose):
        gaps[t], n_active_features[t], \
        n_iters[t] = track_one_path(obj, X, w_init, theta, 
                norm2_X, L_dh,      
                lambdas[t], max_iter, f, tol, screen)

        ws[:, t] = w_init.copy()
        if np.sum(w_init>0.)>=nw_min:
            break

    return ws, lambdas, gaps, n_iters, n_active_features


def spo_nw_min_ex(X, func: int = 0, a: float = 1.0, 
    lambdas: Optional[list] = None, n_lambdas: int = 100, delta: float = 2.0,
    max_iter: int = int(1e5), tol: float = 1e-4, screen: bool = True, f: int = 30, verbose: bool = True,
    nw_min=5):
    """Modified version with first nw_min excluded.

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
    minX = np.min(X)    
    if minX<=0.:
        raise ValueError('The growth rate X must be positive.') 
    X = X - minX
    # Use C-contiguous data to avoid unnecessary memory duplication.
    X = np.ascontiguousarray(X)

    obj = Objective(n_samples, minX, func, a)
    w_init = np.zeros(n_features, dtype=np_type_f)
    theta = obj.comp_dual_var(X, w_init)
    XTtheta = np.dot(X.T, theta)
    
    if lambdas is None:
        lambdas = build_lambdas(XTtheta, n_lambdas, delta)

    n_lambdas = np_type_i(lambdas.shape[0])

    # Useful precomputation
    norm2_X = np.sum(X**2, axis=0)
    L_dh = LA.norm(X * np.sqrt(theta).reshape((-1,1)), ord=2) ** 2
    
    ws = np.zeros((n_features, n_lambdas), dtype=np_type_f)
    gaps = np.ones(n_lambdas, dtype=np_type_f)
    n_active_features = np.zeros(n_lambdas, dtype=np_type_i)
    n_iters = np.zeros(n_lambdas, dtype=np_type_i)

    nw = 0
    for t in tqdm(range(n_lambdas), disable=not verbose):
        gaps[t], n_active_features[t], \
        n_iters[t] = track_one_path(obj, X, w_init, theta, 
                norm2_X, L_dh,      
                lambdas[t], max_iter, f, tol, screen)

        if t==0:
            nw = np.sum(w_init>0.)
            if nw<nw_min:
                X = X[:, w_init==0.] + minX          
                return w_init==0., X

        ws[:, t] = w_init.copy()
        if np.sum(w_init>0.)>=nw_min:
            break
    
    return ws, lambdas, gaps, n_iters, n_active_features    
import numpy as np
import numba as nb
from numba.experimental import jitclass
from .utils import np_type_f, np_type_i, nb_type_f, nb_type_i, FUNC_LOG, FUNC_EXP


@nb.jit(nb_type_f(nb_type_f[:,::1], nb_type_f[::1], nb_type_f), nopython=True, fastmath=True)
def _comp_val_LOG(X, w, eta):
    return - np.mean(np.log(np.dot(X, w) + eta))

@nb.jit(nb_type_f(nb_type_f[::1], nb_type_f), nopython=True, fastmath=True)
def _comp_conj_LOG(theta, eta):
    n = theta.shape[0]
    return np.mean(np.log(n * theta) + 1 - n * eta * theta)

@nb.jit(nb_type_f[::1](nb_type_f[:,::1], nb_type_f[::1], nb_type_f), nopython=True, fastmath=True)
def _comp_dual_var_LOG(X, w, eta):
    return nb_type_f(1.0) / (np.dot(X, w) + eta) / X.shape[0]

@nb.jit(nb_type_f[::1](nb_type_f[::1]), nopython=True, fastmath=True)
def _comp_dual_neg_hess_LOG(theta):
    return 1. / theta**2 / theta.shape[0]


@nb.jit(nb_type_f(nb_type_f[:,::1], nb_type_f[::1], nb_type_f, nb_type_f), nopython=True, fastmath=True)
def _comp_val_EXP(X, w, eta, a):
    return -nb_type_f(1.) + np.mean(np.exp(-(a * np.dot(X, w) + eta)))

@nb.jit(nb_type_f(nb_type_f[::1], nb_type_f, nb_type_f), nopython=True, fastmath=True)
def _comp_conj_EXP(theta, eta, a):
    n = theta.shape[0]
    return - n * np.mean(theta * (np.log(n * theta / a) - 1  + eta)) / a - 1

@nb.jit(nb_type_f[::1](nb_type_f[:,::1], nb_type_f[::1], nb_type_f, nb_type_f), nopython=True, fastmath=True)
def _comp_dual_var_EXP(X, w, eta, a):
    return a * np.exp(- (a * np.dot(X, w) + eta)) / X.shape[0]

@nb.jit(nb_type_f[::1](nb_type_f[::1], nb_type_f, nb_type_f), nopython=True, fastmath=True)
def _comp_dual_neg_hess_EXP(theta, a, lam):
    return lam / a / theta


@nb.jit(nb_type_f[::1](nb_type_f[:,::1], nb_type_f[::1]), nopython=True, fastmath=True)
def _comp_grad(X, theta):
    return - np.dot(X.T, theta)

@nb.jit(nb_type_f[::1](nb_type_f[::1]), nopython=True, fastmath=True)
def _proj(w):
    c = np.sum(w)
    if c>0.:
        w[:] = w / c
    return w

spec = [
    ('eta', nb_type_f),
    ('a', nb_type_f),
    ('L_u', nb_type_f),
    ('L_dH', nb_type_f),
    ('func', nb.int32),
]




@jitclass(spec)
class Objective(object):
    '''
    Empirical negative logarithm function
    F(w) = 1/n * sum_{j=1}^n f(w^T X_j)
    FUNC_LOG :        
        f(x) = - log(eta+x)
    FUNC_EXP :
        f(x) = 1 - exp(-(a*x+eta))
    '''
    def __init__(self, n_samples, eta, func=FUNC_LOG, a=1.):
        '''
        func : int
            0 - log
            1 - minimum variance
        '''
        self.func = func        
        n_samples = n_samples
        if self.func == FUNC_LOG:
            self.a = 0.
            self.eta = eta
            self.L_u = 1./self.eta
            self.L_dH = 1./self.eta**2/n_samples
        else:
            self.a = a
            self.eta = eta
            self.L_u = self.a/np.exp(self.eta)
            self.L_dH = self.a**2/n_samples/np.exp(self.eta)
    

    def val(self, X, w):
        '''
        Params:            
            X: [n,d]
            w: [d,]
        '''         
        if self.func == FUNC_LOG:
            return _comp_val_LOG(X, w, self.eta)
        elif self.func == FUNC_EXP:
            return _comp_val_EXP(X, w, self.eta, self.a)
    
    
    def grad(self, X, w):
        '''The average gradient of F(w) with respect to w.
        Params:
            X: [n,d]
            w: [d,]
        Returns:
            g: [d,]
        '''
        theta = self.comp_dual_var(w, X)
        return _comp_grad(X, theta)
    
    
    def conj(self, theta):  
        ''' F*(theta) = 1/n *  sum_{j=1}^n f*(n * theta_j)
        Params:
            theta: [n,]
        Returns:
            F*: [n,]
        '''
        if self.func == FUNC_LOG:
            return _comp_conj_LOG(theta, self.eta)
        elif self.func == FUNC_EXP:
            return _comp_conj_EXP(theta, self.eta, self.a)
    
    
    def comp_dual_var(self, X, w):
        '''The negative gradient of f(x) with respect to x.
        - lam * theta = 1/n * [f'(w^T X_1), ..., f'(w^T X_n)]^T
        Params:
            X: [n,d]
            w: [d,]            
        Returns:
            theta: [n,]
        '''
        if self.func == FUNC_LOG:
            return _comp_dual_var_LOG(X, w, self.eta)
        elif self.func == FUNC_EXP:
            return _comp_dual_var_EXP(X, w, self.eta, self.a)
    

    def proj(self, w):
        return _proj(w)

    
    def comp_dual_neg_hess(self, theta, lam):
        if self.func == FUNC_LOG:
            return _comp_dual_neg_hess_LOG(theta)
        elif self.func == FUNC_EXP:
            return _comp_dual_neg_hess_EXP(theta, self.a, lam)
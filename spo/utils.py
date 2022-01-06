import numpy as np
from numpy import linalg as LA
import numba as nb

import os 
os.environ['TYPE_FLOAT'] = 'float64'
os.environ['TYPE_INT'] = 'int64'

np_type_f = np.float32 if os.environ['TYPE_FLOAT'] == 'float32' else np.float64
np_type_i = np.int32 if os.environ['TYPE_INT'] == 'int32' else np.int64
nb_type_f = nb.float32 if os.environ['TYPE_FLOAT'] == 'float32' else nb.float64
nb_type_i = nb.int32 if os.environ['TYPE_INT'] == 'int32' else nb.int64


FUNC_LOG = 0
FUNC_EXP = 1


@nb.jit(nb_type_f[::1](nb_type_f[::1], nb_type_f), nopython=True, fastmath=True)
def soft_thresholding(x, lam):
    return np.sign(x) * np.maximum(np.abs(x) - lam, nb_type_f(0.))


@nb.jit(nb_type_f[::1](nb_type_f[::1], nb_type_i, nb_type_f), nopython=True, fastmath=True)
def build_lambdas(XTtheta0, n_lambdas=10, delta=2.0):
    """
    Compute a list of regularization parameters which decrease geometrically.
    Parameters
    ----------
    XTtheta0 : np.array(float)
        The negative gradient evaluated at \(w=0\).
    n_lambdas : int
        The total number of lambdas to build.
    delta : float
        The minimum negative log ratio, which gives the minimum lambda \(\\lambda_{\\max}10^{-\\frac{n_{\\lambda}-1}{n_{\\lambda}} \\delta}\).
    
    Returns
    ----------
    lambdas : np.array(float)
        The list of lambdas.
    """
    lambda_max = LA.norm(XTtheta0, np.inf)

    if n_lambdas == 1:
        lambdas = np.array([lambda_max], dtype=nb_type_f)
    else:        
        lambdas = np.power(10,
            np.linspace(
                np.log10(lambda_max), np.log10(lambda_max)-delta, n_lambdas
                ))
    return lambdas



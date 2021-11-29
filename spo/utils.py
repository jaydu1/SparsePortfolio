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

@nb.jit(nb_type_f(nb_type_f[::1], nb_type_f), nopython=True, fastmath=True)
def epsilon_norm(x, eps):

    """
        Compute the solution in nu of the equation
        sum_i max(|x_i| - (1-eps) * nu, 0)^2 = (eps * nu)^2
    """

    if eps == 1.0:
        return LA.norm(x)
    elif eps == 0.0:
        return np.max(np.abs(x))
    
    alpha, R = 1-eps, eps
    
    len_x = x.shape[0]
    R2 = R * R
    alpha2 = alpha * alpha
    delta = 0.
    R2onalpha2 = R2 / alpha2
    alpha2j0 = 0.
    j0alpha2_R2 = 0.
    alpha_S = 0.
    S = 0.
    S2 = 0.
    a_k = 0.
    b_k = 0.
    norm_inf = np.max(x)
    ratio_ = alpha * (norm_inf)

    if norm_inf == 0:
        return 0
    
    n_I = 0
    zx = np.zeros(len_x, dtype=nb_type_f)
    for k in range(len_x):
        if np.abs(x[k]) > ratio_:
            zx[n_I] = np.abs(x[k])
            n_I += 1

    zx = np.sort(zx)[::-1]

    if norm_inf == 0:
        return 0

    if n_I == 1:
        return zx[0]

    for k in range(n_I - 1):

        S += zx[k]
        S2 += zx[k] * zx[k]
        b_k = S2 / (zx[k + 1] * zx[k + 1]) - 2 * S / zx[k + 1] + k + 1

        if a_k <= R2onalpha2 and R2onalpha2 < b_k:
            j0 = k + 1
            break
    else:
        j0 = n_I
        S += zx[n_I - 1]
        S2 += zx[n_I - 1] * zx[n_I - 1]

    alpha_S = alpha * S
    alpha2j0 = alpha2 * j0

    if (alpha2j0 == R2):
        return S2 / (2 * alpha_S)

    j0alpha2_R2 = alpha2j0 - R2
    delta = alpha_S * alpha_S - S2 * j0alpha2_R2

    return (alpha_S - np.sqrt(delta)) / j0alpha2_R2


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



import pandas as pd
import numpy as np
from spo import spo_l1_path, spo_nw_min
import sys
import os
import time
import sys
sys.path.append("./")

df = pd.read_csv('data/nyse/NYSE.csv', index_col=[0])
X = df.values
X = X / np.max(np.abs(X))
X = X + 1.

n_lambdas = 100
func_names = ['LOG', 'EXP']

def select_delta(X, func, a, nw_min=1, nw_max=50):
    delta = 3.0
    n_w = 0 

    while True and delta<8.:
        _, lambdas, _, _, _ = spo_l1_path(X, func, a, None, 2, max_iter=int(0), delta=delta, verbose=False)
        ws, _, _, _, _ = spo_l1_path(X, func, a, lambdas[-1:], None, screen=True, max_iter=int(1e6), f=200, tol=1e-8, verbose=False)
        n_w = np.sum(ws>0.)
        if n_w>=nw_max:
            break
        delta += 0.5
    delta_max = delta
    lambda_max = lambdas[0]
    lambdas = np.power(10,
        np.linspace(
            np.log10(lambda_max), np.log10(lambda_max)-delta_max,  200
            ))
    ws, _, gaps, _, _ = spo_nw_min(X, func, a, lambdas, None, 
        screen=True, max_iter=int(1e6), f=200, tol=1e-12, verbose=True, nw_min=nw_max)
    n_ws = np.sum(ws>0., axis=0)
    lambdas = lambdas[n_ws>=nw_min]
    ws = ws[:, (n_ws>=nw_min)]
    gaps = [n_ws>=nw_min]
    return ws, lambdas, gaps


from joblib import Parallel, delayed
with Parallel(n_jobs=-1, verbose=100) as parallel:
    ws_list = []
    lambdas_list = []
    gaps_list = []
    times = []
    for func in range(2):
        if func == 0:
            a_list = [1.]
            nw_max = 50
        else:
            a_list = [5e-2,1e-1,5e-1,1.,1.5]
            nw_max = 25
        for a in a_list:
            t1 = time.time()
            ws, lambdas, gaps = select_delta(X, func, a, nw_min=1, nw_max=nw_max)
            times.append(time.time() - t1)
            ws_list.append(ws)
            gaps_list.append(gaps)
            lambdas_list.append(lambdas)

    ws_list = np.array(ws_list)
    lambdas_list = np.array(lambdas_list)
    np.savez('result/res_spo_efficient_frontier.npz',                         
                ws_list=ws_list,
                lambdas_list=lambdas_list,
                gaps_list=gaps_list,
                times=times
            )
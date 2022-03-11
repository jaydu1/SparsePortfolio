from typing import Optional

import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import sys
sys.path.append("./")

df = pd.read_csv('data/nyse/NYSE.csv', index_col=[0])
X = df.values
X = X/np.max(np.abs(X))
X += 1

from spo import spo_l1_path
import time
n_lambdas = 10
func_names = ['LOG', 'EXP']
a_list = [[1.], [1.]] # , 5e-2,1e-1,5e-1,1.5

n_tols = 100
tols = np.flip(10**(np.linspace(-6, 0, n_tols+1)[:-1]))

for func in range(2):
    res = {}
    for a in a_list[func]:
        if func==0:
            name = func_names[func]
            delta = 2.
        else:
            name = func_names[func] + '-%.2f'%a
            delta = 1.0
            
        
        
        _, lambdas, gap, _, _ = spo_l1_path(X, func, a, None, n_lambdas+1, screen=False, delta=delta, max_iter=int(0), tol=1e-4, verbose=False) 
        lam_max = lambdas[0]
        ratio_lam = np.array([5e-1])
        lambdas = lam_max*ratio_lam
        for i_lam,lam in enumerate(lambdas):
            print(func, lam)
            n_iters1 = 0
            n_iters2 = 0
            time_screened = np.zeros(n_tols, dtype=float)
            time_screened.fill(np.nan)
            time_unscreened = np.zeros(n_tols, dtype=float)
            time_unscreened.fill(np.nan)
            for j in tqdm(range(n_tols)):
                tol = tols[j]
                times = []
                
                if n_iters1<int(1e6)-1:
                    t1 = time.time()
                    _, _, gap, n_iters1, _ = spo_l1_path(X, func, a, np.array([lam]), None, screen=True, f=200, max_iter=int(1e6), tol=tol, verbose=False) 
                    time_screened[j] = time.time() - t1

                if n_iters2<int(1e6)-1:
                    t1 = time.time()
                    _, _, gap, n_iters2, _ = spo_l1_path(X, func, a, np.array([lam]), None, screen=False, f=200, max_iter=int(1e6), tol=tol, verbose=False) 
                    time_unscreened[j] = time.time() - t1
    
            np.savez('result/res_ex3_%s_%.2f.npz'%(func_names[func], ratio_lam[i_lam]), tols=tols, time_screened=time_screened, time_unscreened=time_unscreened)
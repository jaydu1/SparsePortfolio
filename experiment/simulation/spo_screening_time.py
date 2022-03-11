from typing import Optional

import pandas as pd
import numpy as np
import time
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
a_list = [[1.], [5e-2,1e-1,5e-1,1.,1.5]]

n_lambdas = 50
for func in range(2):
    res = {}
    for a in a_list[func]:
        if func==0:
            name = func_names[func]
            delta = 2.
        else:
            name = func_names[func] + '-%.2f'%a
            delta = 1.0
            
        res = []
        
        _, lambdas, _, _, _ = spo_l1_path(X, func, a, None, n_lambdas+1, screen=False, delta=delta, max_iter=int(0), tol=1e-4, verbose=False) 
        lam_max = lambdas[0]

        for lam in lambdas[1:]:
            times = []
            for is_screen in [True, False]:
                t1 = time.time()
                _, _, gaps, _, _ = spo_l1_path(X, func, a, np.array([lam]), 2, screen=is_screen, f=200, max_iter=int(1e6), tol=1e-4, verbose=False) 
                times.append(time.time() - t1)

            res.append(times[0]/times[1])
    
    np.savez('result/simulation/res_ex2_%s.npz'%func_names[func], res=res, lambdas=lambdas)
import pandas as pd
import numpy as np
import sys
import os


func_names = ['LOG', 'EXP']

n_lambdas = 100
n_folds = 10
n_days_train = 120
n_days_hold = 63

n_ws = np.arange(1,51)

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

def fit(i, df, func, a):
    from spo import spo_l1_path
    i_len = np.minimum((i + 1) * n_days_hold, len(test_date)) - i * n_days_hold
    idx = id_recal[i]

    score_test = np.zeros((i_len, len(n_ws)))
    ws_test = np.zeros((1, d, len(n_ws)))

    X = df.iloc[idx-n_days_train:idx,:].values.copy()
    X_test = df_hold.iloc[idx:idx+n_days_hold,:].values.copy()

    ws, _, _, _, _ = spo_l1_path(X, func, a, None, n_lambdas, screen=True, max_iter=int(1e5), f=30, tol=1e-7, verbose=False)
    
    print(np.sum(ws > 0, axis=0))
    for j, n_w in enumerate(n_ws):
        id_lams = np.where((np.sum(ws > 0, axis=0) > 0) & (np.sum(ws > 0, axis=0) <= n_w))[0]
        if len(id_lams) > 0:
            w = ws[:, id_lams[-1]]
            score_test[:, j] = np.dot(X_test, w) - 1.
            ws_test[:, :, j] = w
        else:
            score_test[:, j] = 0.
            ws_test[:, :, j] = 0.
    return score_test, ws_test 

from joblib import Parallel, delayed
with Parallel(n_jobs=-1, verbose=100) as parallel:
    for func in range(2):
        if func == 0:
            a_list = [1.]
        else:
            a_list = [5e-2,1e-1,5e-1,1.,1.5]
        for a in a_list:
            out = parallel(delayed(fit)(i, df, func, a) for i in range(len(id_recal)))
            score_test_list, ws_test_list = zip(*out)
            score_test_list = np.concatenate(score_test_list, axis=0)
            ws_test_list = np.concatenate(ws_test_list, axis=0)
            np.savez('result/res_%s_%d_%d_%.2f_n.npz'%(func_names[func],n_days_train,n_days_hold,a), 
                        score_test_list=score_test_list, 
                        ws_test_list=ws_test_list,
                        test_date=test_date,
                        n_ws=n_ws
                    )
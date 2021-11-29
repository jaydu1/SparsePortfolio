import pandas as pd
import numpy as np
import time 

func_names = ['LOG', 'EXP']

n_ws = np.arange(1,51)

n_lambdas = 100
n_folds = 10

df_close = pd.read_csv('data/russell/PRC_russell.csv', index_col=[0])
df_open = pd.read_csv('data/russell/OPENPRC_russell.csv', index_col=[0])
df = df_close/df_open

T, d = df.shape

id_begin = np.where(df.index>=20050101)[0][0]
df_hold = pd.read_csv('data/russell/RET_russell.csv', index_col=[0]) + 1

id_recal = []
id_train_begin = []
id_codes_list = []
id_test_end = []
for year in range(2005,2021):
    for month in [1,7]:#range(1,13):
        date = int('%d%02d01'%(year,month))        
        date_train = int('%d%02d01'%(year-1,month))
        codes = np.array(df.columns)

        _df = df[codes].iloc[(df.index>=date_train)&(df.index<date)].copy()
        codes = codes[
            ~np.any(np.isnan(_df), axis=0)
        ]
        id_codes_list.append(
            np.array([np.where(np.array(list(df.columns))==i)[0][0] for i in codes])
            )
        id_recal.append(np.where(df.index>=date)[0][0])
        id_train_begin.append(np.where(df.index>=date_train)[0][0])
        id_test_end.append(np.where(df.index<int('%d%02d01'%(year+int(month==7),(month+6)%12)))[0][-1])
        

n_recal = len(id_recal)

df = df.fillna(1.)
df_hold.iloc[id_recal,:] = df.iloc[id_recal,:].copy()
df_hold = df_hold.fillna(1.)
test_date = np.array(df.index[id_begin:id_test_end[-1]+1])

def fit(i, df, func, a):
    from spo import spo_l1_path
    score_test = np.zeros((id_test_end[i] - id_recal[i] + 1,len(n_ws)))
    ws_test = np.empty((1, d, len(n_ws)))
    ws_test.fill(np.nan)
    ws_test_all = np.empty((1, d, n_lambdas))
    ws_test_all.fill(np.nan)

    X = df.iloc[id_train_begin[i]:id_recal[i],:].values[:,id_codes_list[i]].copy()
    X_test = df_hold.iloc[id_recal[i]:id_test_end[i]+1,:].values[:,id_codes_list[i]].copy()

    t1 = time.time()
    ws, lambdas, _, _, _ = spo_l1_path(X, func, a, None, n_lambdas, delta=4.,
        screen=True, max_iter=int(1e5), f=200, tol=1e-8)    
    time_elapsed = time.time() - t1
    
    print(np.sum(ws > 0, axis=0))
    for j, n_w in enumerate(n_ws):
        id_lams = np.where((np.sum(ws > 0, axis=0) > 0) & (np.sum(ws > 0, axis=0) <= n_w))[0]
        if len(id_lams) > 0:
            w = ws[:, id_lams[-1]]
            score_test[:, j] = np.dot(X_test, w) - 1.
            ws_test[0, id_codes_list[i], j] = w
        else:
            ws_test[0, id_codes_list[i], j] = 0.
    ws_test_all[0, id_codes_list[i], :] = ws
    return score_test, ws_test, lambdas, time_elapsed, ws_test_all

from joblib import Parallel, delayed
with Parallel(n_jobs=-1, verbose=100) as parallel:
    for func in range(2):
        if func == 0:
            a_list = [1.]
        else:
            a_list = [5e-2,1e-1,5e-1,1.,1.5]
        for a in a_list:
            print(func,a)
            out = parallel(delayed(fit)(i, df, func, a) for i in range(len(id_recal)))
            score_test_list, ws_test_list, lambdas, time_elapsed, ws_test_all = zip(*out)
            score_test_list = np.concatenate(score_test_list, axis=0)
            ws_test_list = np.concatenate(ws_test_list, axis=0)
            lambdas = np.concatenate(lambdas, axis=0)
            time_elapsed = time_elapsed = np.array(time_elapsed)
            ws_test_all = np.concatenate(ws_test_all, axis=0)
            np.savez('result/res_russell_%s_%.2f_n.npz'%(func_names[func],a), 
                        score_test_list=score_test_list, 
                        ws_test_list=ws_test_list,
                        test_date=test_date,
                        n_ws=n_ws,
                        lambdas=lambdas,
                        time=time_elapsed,
                        ws_test_all=ws_test_all
                    )
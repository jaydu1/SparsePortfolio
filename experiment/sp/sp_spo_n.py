import pandas as pd
import numpy as np
import sys
import os

sys.path.append("./")
path_data = './data/sp/'
path_result = 'result/'
os.makedirs(path_result, exist_ok=True)

func_names = ['LOG', 'EXP']
k = int(sys.argv[1])
if k == 0:
    func = 0
    a = 1.
else:
    func = 1
    a_list = [5e-2,1e-1,5e-1,1.,1.5]
    a = a_list[k-1]

n_lambdas = 100
n_days_train = 120
n_days_hold = 63

n_ws = np.arange(1,51)

df_close = pd.read_csv(path_data+'sp500_PRC.csv', index_col=[0])
df_open = pd.read_csv(path_data+'sp500_OPENPRC.csv', index_col=[0])
df = df_close/df_open
T, d = df.shape

id_begin = np.where(df.index>=20110101)[0][0]
df_hold = pd.read_csv(path_data+'sp500_RET.csv', index_col=[0]) + 1
id_recal = np.arange(id_begin, len(df.index), n_days_hold)
df_hold.iloc[id_recal,:] = df.iloc[id_recal,:].copy()
df_hold = df_hold.fillna(1.)
n_recal = len(id_recal)

test_date = np.array(df.index[id_begin:])

df_listed = pd.read_csv(path_data+'sp500_listed.csv', index_col=[0])
df_listed.index = np.array(pd.to_datetime(df_listed.index).strftime('%Y%m%d')).astype(int)
id_codes_list = []
for idx in id_recal:
    codes = np.array(
        df_listed.columns[
        (np.all(df_listed[
            (df_listed.index>=df.index[idx-n_days_hold*4])&
            (df_listed.index<=df.index[idx])]==1, axis=0))])
    codes = codes[~df.iloc[idx-n_days_train:idx,:].loc[:,codes].isnull().any()]
    id_codes_list.append(
        np.sort([np.where(np.array(list(df.columns))==i)[0][0] for i in codes])
        )

def fit(i, df, func, a):
    from spo import spo_l1_path
    i_len = np.minimum((i + 1) * n_days_hold, len(test_date)) - i * n_days_hold
    idx = id_recal[i]

    score_test = np.zeros((i_len, len(n_ws)))
    ws_test = np.empty((1, d, len(n_ws)))
    ws_test.fill(np.nan)
    X = df.iloc[idx-n_days_train:idx,:].values[:,id_codes_list[i]].copy()
    clip = 2.5
    clip_low, clip_high = np.percentile(
        X.flatten(), [clip, 100.0 - clip]
    )
    X = np.clip(X, clip_low, clip_high)
    X_test = df_hold.iloc[idx:idx+n_days_hold,:].values[:,id_codes_list[i]].copy()
    
    delta = 2.
    ws, _, _, _, _ = spo_l1_path(X, func, a, None, n_lambdas, delta=delta,
        screen=True, max_iter=int(1e5), f=30, tol=1e-7, verbose=False)
    
    print(np.sum(ws > 0, axis=0))
    for j, n_w in enumerate(n_ws):
        id_lams = np.where((np.sum(ws > 0, axis=0) > 0) & (np.sum(ws > 0, axis=0) <= n_w))[0]
        if len(id_lams) > 0:
            w = ws[:, id_lams[-1]]
            score_test[:, j] = np.dot(X_test, w) - 1.
            ws_test[0, id_codes_list[i], j] = w
        else:
            score_test[:, j] = 0.
            ws_test[0, id_codes_list[i], j] = 0.
    return score_test, ws_test 

from joblib import Parallel, delayed
with Parallel(n_jobs=-1, verbose=100) as parallel:
    out = parallel(delayed(fit)(i, df, func, a) for i in range(len(id_recal)))
    score_test_list, ws_test_list = zip(*out)
    score_test_list = np.concatenate(score_test_list, axis=0)
    ws_test_list = np.concatenate(ws_test_list, axis=0)
    np.savez(path_result+'res_sp_%s_%.2f_n.npz'%(func_names[func],a), 
                score_test_list=score_test_list, 
                ws_test_list=ws_test_list,
                test_date=test_date,
                n_ws=n_ws
            )
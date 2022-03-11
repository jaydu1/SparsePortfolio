import pandas as pd
import numpy as np
import sys
import os

sys.path.append("./")
path_data = './data/russell/'
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
n_ws = np.arange(1,51)

df_close = pd.read_csv(path_data+'russell2000_PRC.csv', index_col=[0])
df_open = pd.read_csv(path_data+'russell2000_OPENPRC.csv', index_col=[0])
df = df_close/df_open
T, d = df.shape

id_begin = np.where(df.index>=20050101)[0][0]
df_hold = pd.read_csv(path_data+'russell2000_RET.csv', index_col=[0]) + 1
df_listed = pd.read_csv(path_data+'russell2000_listed.csv', index_col=[0])

id_recal = []
id_train_begin = []
id_codes_list = []
id_test_end = []
for year in range(2005,2021):
    for month in [1,7]:
        date = int('%d%02d01'%(year,month))        
        date_train = int('%d%02d01'%(year-1,month))
        codes = df_listed.columns[
            (np.all(df_listed[
                (df_listed.index>=int('%d%02d01'%(year-int(month==1),(month+6)%12)))&
                (df_listed.index<=date)]==1, axis=0))]
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
    clip = 2.5
    clip_low, clip_high = np.percentile(
        X.flatten(), [clip, 100.0 - clip]
    )
    X = np.clip(X, clip_low, clip_high)
    X_test = df_hold.iloc[id_recal[i]:id_test_end[i]+1,:].values[:,id_codes_list[i]].copy()
    
    delta = 4.
    ws, _, _, _, _ = spo_l1_path(X, func, a, None, n_lambdas, delta=delta,
        screen=True, max_iter=int(1e5), f=200, tol=1e-8, verbose=False)
    
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
    ws_test_all[0, id_codes_list[i], :] = ws
    return score_test, ws_test, ws_test_all

from joblib import Parallel, delayed
with Parallel(n_jobs=-1, verbose=100) as parallel:
    out = parallel(delayed(fit)(i, df, func, a) for i in range(len(id_recal)))
    score_test_list, ws_test_list = zip(*out)
    score_test_list = np.concatenate(score_test_list, axis=0)
    ws_test_list = np.concatenate(ws_test_list, axis=0)
    np.savez(path_result+'res_russell_%s_%.2f_n.npz'%(func_names[func],a), 
                score_test_list=score_test_list, 
                ws_test_list=ws_test_list,
                test_date=test_date,
                n_ws=n_ws,
                ws_test_all=ws_test_all
            )

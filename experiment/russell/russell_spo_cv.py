import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
import time
import os
import sys

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
n_folds = 10

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

delta = 3.
def fit(i, df, func, a):
    from spo import spo_l1_path, spo_nw_min
    
    score_test = np.zeros((id_test_end[i] - id_recal[i] + 1,))
    ws_test = np.empty((1, d))
    ws_test.fill(np.nan)

    X = df.iloc[id_train_begin[i]:id_recal[i],:].values[:,id_codes_list[i]].copy()
    clip = 2.5
    clip_low, clip_high = np.percentile(
        X.flatten(), [clip, 100.0 - clip]
    )
    X = np.clip(X, clip_low, clip_high)
    
    X_test = df_hold.iloc[id_recal[i]:id_test_end[i]+1,:].values[:,id_codes_list[i]].copy()
    
    t1 = time.time()
    kf = TimeSeriesSplit(n_splits=n_folds, test_size=None)

    _, lambdas, _, _, _ = spo_l1_path(X, func, a, None, n_lambdas, max_iter=int(0), delta=delta, verbose=False)

    score_val = np.zeros((n_folds, n_lambdas))
    n_val = np.zeros((n_folds, n_lambdas))
    for j, (train_index, val_index) in tqdm(enumerate(kf.split(X)), total = n_folds):
        X_train, X_val = X[train_index], X[val_index]

        ws, _, _, _, _ = spo_l1_path(X_train, func, a, lambdas, None,
            screen=True, max_iter=int(1e4), f=200, tol=1e-5, verbose=False)

        # by returns
        # score_val[j] = np.where(np.sum(ws>0, axis=0)>0, np.mean(X_val @ ws, axis=0), -np.inf)

        # by sharpe ratio
        ret = np.log((X_val-1) @ ws + 1)
        sd = np.std(ret, axis=0)
        score_val[j] = np.divide(np.mean(ret, axis=0), sd, out=np.zeros_like(sd), where=sd != 0)
        n_val[j] = np.sum(ws>0., axis=0)
    
    id_lam = np.argmax(np.median(score_val, axis=0))
    lambdas = lambdas[id_lam:id_lam+5]        
    ws, _, _, _, _ = spo_nw_min(X, func, a, lambdas, None, 
                                 screen=True, max_iter=int(1e5), f=200, tol=1e-8, nw_min=1)
    id_lam = np.where(np.sum(ws>0, axis=0) > 0)[0]
    time_elapsed = time.time() - t1
    if len(id_lam)>0:
        w = ws[:, id_lam[0]]    
        score_test = np.dot(X_test, w) - 1
        ws_test[0, id_codes_list[i]] = w        
    else:
        ws_test[0, id_codes_list[i]] = 0.   

    ret = np.log(score_test+1)
    print(i, np.sum(ws>0, axis=0), np.nancumprod(score_test+1)[-1], np.mean(ret)/np.std(ret))
    return score_test, ws_test, lambdas, time_elapsed


from joblib import Parallel, delayed

with Parallel(n_jobs=-1, verbose=100) as parallel:
    out = parallel(delayed(fit)(i, df, func, a) for i in range(len(id_recal)))
    score_test_list, ws_test_list, lambdas, time_elapsed = zip(*out)
    score_test_list = np.concatenate(score_test_list, axis=0)
    ws_test_list = np.concatenate(ws_test_list, axis=0)
    lambdas = np.concatenate(lambdas, axis=0)
    time_elapsed = np.array(time_elapsed)
    np.savez(path_result+'res_russell_%s_%.2f.npz'%(func_names[func],a), 
                        score_test_list=score_test_list, 
                        ws_test_list=ws_test_list,
                        test_date=test_date,
                        lambdas=lambdas,
                        time=time_elapsed                        
                    )
    ret = np.log(score_test_list+1)
    print(func, a, np.nancumprod(score_test_list+1)[-1], np.mean(ret)/np.std(ret))
        

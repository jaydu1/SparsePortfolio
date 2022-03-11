import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import KFold
from tqdm import tqdm
import os
import sys

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
n_folds = 5
n_days_train = 120
n_days_hold = 63

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
    from spo import spo_l1_path, spo_nw_min
    i_len = np.minimum((i + 1) * n_days_hold, len(test_date)) - i * n_days_hold
    idx = id_recal[i]

    score_test = np.zeros((i_len,))
    ws_test = np.empty((1, d))
    ws_test.fill(np.nan)

    X = df.iloc[idx-n_days_train:idx,:].values[:,id_codes_list[i]].copy()
    X_test = df_hold.iloc[idx:idx+n_days_hold,:].values[:,id_codes_list[i]].copy()
    
    delta = 2.
    _, lambdas, _, _, _ = spo_l1_path(X, func, a, None, n_lambdas, screen=True, delta=delta,
                                          max_iter=int(0), verbose=False)

    kf = TimeSeriesSplit(n_splits=n_folds, test_size=None)
    #kf = KFold(n_splits=n_folds)
    score_val = np.zeros((n_folds, n_lambdas))
    for j, (train_index, val_index) in tqdm(enumerate(kf.split(X)), total = n_folds):
        X_train, X_val = X[train_index], X[val_index]

        ws, _, _, _, _ = spo_l1_path(X_train, func, a, lambdas, None, 
                                     screen=True, max_iter=int(1e4), f=30, tol=1e-5, verbose=False)

        # by returns
        # score_val[j] = np.where(np.sum(ws>0, axis=0)>0, np.mean(X_val @ ws, axis=0), -np.inf)

        # by sharpe ratio
        ret = np.log((X_val-1) @ ws + 1)
        sd = np.std(ret, axis=0)
        score_val[j] = np.divide(np.mean(ret, axis=0), sd, out=np.zeros_like(sd), where=sd != 0)

    id_lam = np.argmax(np.median(score_val, axis=0))    
    lambdas = lambdas[id_lam:id_lam+5]        
    ws, _, _, _, _ = spo_nw_min(X, func, a, lambdas, None, 
                                 screen=True, max_iter=int(1e5), f=30, tol=1e-8, nw_min=1)
    id_lams = np.where(np.sum(ws>0, axis=0) > 0)[0]
    
    if len(id_lams)>0:
        w = ws[:, id_lams[0]]
        score_test = np.dot(X_test, w) - 1
        ws_test[0, id_codes_list[i]] = w
    else:
        score_test = [0.]
        ws_test[0, id_codes_list[i]] = 0.

    print(i, np.sum(ws>0, axis=0), np.cumprod(score_test+1.))
    return score_test, ws_test, lambdas

from joblib import Parallel, delayed
with Parallel(n_jobs=-1, verbose=100) as parallel:
    out = parallel(delayed(fit)(i, df, func, a) for i in range(len(id_recal)))
    score_test_list, ws_test_list, lambdas = zip(*out)
    score_test_list = np.concatenate(score_test_list, axis=0)
    ws_test_list = np.concatenate(ws_test_list, axis=0)
    lambdas = np.concatenate(lambdas, axis=0)
    np.savez(path_result+'res_sp_%s_%.2f.npz'%(func_names[func],a), 
                score_test_list=score_test_list, 
                ws_test_list=ws_test_list,
                test_date=test_date,
                lambdas=lambdas
            )
    ret = np.log(score_test_list+1)
    print(func, a, np.nancumprod(score_test_list+1)[-1], np.mean(ret)/np.std(ret))
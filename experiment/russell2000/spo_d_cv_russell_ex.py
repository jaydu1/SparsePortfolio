import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
import time

func_names = ['LOG', 'EXP']

n_lambdas = 100
n_folds = 10

df_close = pd.read_csv('data/PRC_russell.csv', index_col=[0])
df_open = pd.read_csv('data/OPENPRC_russell.csv', index_col=[0])
df = df_close/df_open

T, d = df.shape

id_begin = np.where(df.index>=20050101)[0][0]
df_hold = pd.read_csv('data/RET_russell.csv', index_col=[0]) + 1

id_recal = []
id_train_begin = []
id_codes_list = []
id_test_end = []
for year in range(2005,2021):
    for month in [1,7]:#range(1,13):
        date = int('%d%02d01'%(year,month))        
        date_train = int('%d%02d01'%(year-1,month))
        codes = np.array(df.columns)
        # date_train = int('%d%02d01'%(year-int(month<=6),(month-7)%12+1))

        _df = df[codes].iloc[(df.index>=date_train)&(df.index<date)].copy()
        codes = codes[
            ~np.any(np.isnan(_df), axis=0)
            # (~np.any(np.isnan(_df).rolling(window=21, axis=0).sum() >= 21, axis=0)) &
            # (np.isnan(_df).rolling(window=21, axis=0).sum().iloc[-1,:] == 0)
        ]
        id_codes_list.append(
            np.array([np.where(np.array(list(df.columns))==i)[0][0] for i in codes])
            )
        id_recal.append(np.where(df.index>=date)[0][0])
        id_train_begin.append(np.where(df.index>=date_train)[0][0])
        # id_test_end.append(np.where(df.index<int('%d%02d01'%(year+int(month==12),month%12+1)))[0][-1])
        # id_test_end.append(np.where(df.index<int('%d%02d01'%(year+int(month==11),(month+1)%12+1)))[0][-1])
        id_test_end.append(np.where(df.index<int('%d%02d01'%(year+int(month==7),(month+6)%12)))[0][-1])
        

n_recal = len(id_recal)

df = df.fillna(1.)
df_hold.iloc[id_recal,:] = df.iloc[id_recal,:].copy()
df_hold = df_hold.fillna(1.)
test_date = np.array(df.index[id_begin:id_test_end[-1]+1])


def fit(i, df, func, a):
    from spo import spo_l1_path, spo_nw_min, spo_nw_min_ex
    
    score_test = np.zeros((id_test_end[i] - id_recal[i] + 1,))
    ws_test = np.empty((1, d))
    ws_test.fill(np.nan)

    X = df.iloc[id_train_begin[i]:id_recal[i],:].values[:,id_codes_list[i]].copy()
    X_test = df_hold.iloc[id_recal[i]:id_test_end[i]+1,:].values[:,id_codes_list[i]].copy()
    
    t1 = time.time()
    kf = TimeSeriesSplit(n_splits=n_folds, test_size=None)

    _, lambdas, _, _, _ = spo_l1_path(X, func, a, None, n_lambdas, max_iter=int(0), delta=3., verbose=False)

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
    
    score_val[:, np.mean(n_val, axis=0)<nw_min] = - np.inf
    id_lam = np.argmax(np.median(score_val, axis=0))
    lambdas = lambdas[id_lam:]
    out = spo_nw_min_ex(X, func, a, lambdas, None, screen=True, max_iter=int(1e5), f=200, tol=1e-8, nw_min=nw_min)
    if len(out)==2:
        id_ws, X = out

        kf = TimeSeriesSplit(n_splits=n_folds, test_size=None)

        _, lambdas, _, _, _ = spo_l1_path(X, func, a, None, n_lambdas, max_iter=int(0), delta=3., verbose=False)

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
        
        score_val[:, np.mean(n_val, axis=0)<nw_min] = - np.inf
        id_lam = np.argmax(np.median(score_val, axis=0))
        lambdas = lambdas[id_lam:]
        ws = np.zeros((X_test.shape[1], len(lambdas)))
        ws[id_ws, :], _, _, _, _ = spo_nw_min(X, func, a, lambdas, None, screen=True, max_iter=int(1e5), f=200, tol=1e-8, nw_min=nw_min)
    else:
        ws, _, _, _, _ = out

    id_lam = np.where(np.sum(ws>0., axis=0) >= nw_min)[0][-1]
    time_elapsed = time.time() - t1

    w = ws[:, id_lam]    
    score_test = np.dot(X_test, w) - 1
    ws_test[0, id_codes_list[i]] = w
    time_elapsed = time.time() - t1  

    ret = np.log(score_test+1)
    print(i, np.sum(ws>0, axis=0), np.nancumprod(score_test+1)[-1], np.mean(ret)/np.std(ret))
    return score_test, ws_test, lambdas, time_elapsed

nw_min = 5

from joblib import Parallel, delayed
with Parallel(n_jobs=-1, verbose=100) as parallel:
    for func in range(2):
        if func == 0:
            a_list = [1.]
        else:
            a_list = [5e-2,1e-1,5e-1,1.,1.5]
        for a in a_list:
            out = parallel(delayed(fit)(i, df, func, a) for i in range(len(id_recal)))
            score_test_list, ws_test_list, lambdas, time_elapsed = zip(*out)
            score_test_list = np.concatenate(score_test_list, axis=0)
            ws_test_list = np.concatenate(ws_test_list, axis=0)
            lambdas = np.concatenate(lambdas, axis=0)
            time_elapsed = np.array(time_elapsed)
            np.savez('result/res_russell_%d_%s_%.2f_ex.npz'%(nw_min, func_names[func],a), 
                        score_test_list=score_test_list, 
                        ws_test_list=ws_test_list,
                        test_date=test_date,
                        lambdas=lambdas,
                        time=time_elapsed                        
                    )
            print(score_test_list)
            ret = np.log(score_test_list+1)
            print(func, a, np.nancumprod(score_test_list+1)[-1], np.mean(ret)/np.std(ret))
        
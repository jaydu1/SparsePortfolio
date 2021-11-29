import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

func_names = ['LOG', 'EXP']

n_lambdas = 100
n_folds = 10
n_days_train = 120
n_days_hold = 63

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

    score_test = np.zeros((i_len,))
    ws_test = np.zeros((1, d))

    X = df.iloc[idx-n_days_train:idx,:].values.copy()
    X_test = df_hold.iloc[idx:idx+n_days_hold,:].values.copy()
    
    kf = TimeSeriesSplit(n_splits=n_folds, test_size=None)

    _, lambdas, _, _, _ = spo_l1_path(X, func, a, None, n_lambdas, screen=True, max_iter=int(0), f=20, tol=1e-4, verbose=False)

    score_val = np.zeros((n_folds, n_lambdas))
    for j, (train_index, val_index) in tqdm(enumerate(kf.split(X)), total = n_folds):
        X_train, X_val = X[train_index], X[val_index]

        ws, _, _, _, _ = spo_l1_path(X_train, func, a, lambdas, None, screen=True, max_iter=int(1e4), f=30, tol=1e-5, verbose=False)

        # by returns
        # score_val[j] = np.where(np.sum(ws>0, axis=0)>0, np.mean(X_val @ ws, axis=0), -np.inf)

        # by sharpe ratio
        ret = np.log((X_val-1) @ ws + 1)
        sd = np.std(ret, axis=0)
        score_val[j] = np.divide(np.mean(ret, axis=0), sd, out=np.zeros_like(sd), where=sd != 0)

    id_lam = np.argmax(np.median(score_val, axis=0))
    lambdas = lambdas[id_lam:id_lam+5]
    ws, _, _, _, _ = spo_l1_path(X, func, a, lambdas, None, screen=True, max_iter=int(1e5), f=30, tol=1e-8)
    id_lams = np.where(np.sum(ws>0, axis=0) > 0)[0]
    print(np.sum(ws>0, axis=0))
    if len(id_lams)>0:
        w = ws[:, id_lams[0]]
        score_test = np.dot(X_test, w) - 1
        ws_test[0, :] = w
    else:
        score_test = 0.
        ws_test[0, :] = 0.

    return score_test, ws_test, lambdas

from joblib import Parallel, delayed
with Parallel(n_jobs=-1, verbose=100) as parallel:
    for func in range(2):
        if func == 0:
            a_list = [1.]
        else:
            a_list = [5e-2,1e-1,5e-1,1.,1.5]
        for a in a_list:
            out = parallel(delayed(fit)(i, df, func, a) for i in range(len(id_recal)))
            score_test_list, ws_test_list, lambdas = zip(*out)
            score_test_list = np.concatenate(score_test_list, axis=0)
            ws_test_list = np.concatenate(ws_test_list, axis=0)
            lambdas = np.concatenate(lambdas, axis=0)
            np.savez('result/res_%s_%d_%d_%.2f.npz'%(func_names[func],n_days_train,n_days_hold,a), 
                        score_test_list=score_test_list, 
                        ws_test_list=ws_test_list,
                        test_date=test_date,
                        lambdas=lambdas
                    )
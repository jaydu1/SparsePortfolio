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
id_train_begin = []
id_test_end = []
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
    id_train_begin.append(np.where(df.index==df.iloc[idx-n_days_train:idx,:].index[0])[0][-1])
    id_test_end.append(np.where(df.index==df.iloc[idx:idx+n_days_hold,:].index[-1])[0][-1])


score_test_list = []
ws_test_list = np.zeros((len(id_recal), d))
ws_test_list.fill(np.nan)
for i in range(len(id_recal)):
    X_test = df_hold.iloc[id_recal[i]:id_test_end[i]+1,:].values[:,id_codes_list[i]].copy()
    score_test_list.append(np.mean(X_test, axis=1))
    ws_test_list[i, id_codes_list[i]] = 1./len(id_codes_list[i])
score_test_list = np.concatenate(score_test_list) - 1.

# equally weighted portfolio
np.savez('result/res_sp_EW.npz', 
            score_test_list=score_test_list, 
            ws_test_list=ws_test_list,
            test_date=test_date
        )
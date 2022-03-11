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

score_test_list = []
ws_test_list = np.zeros((len(id_recal), d))
ws_test_list.fill(np.nan)
for i in range(len(id_recal)):
    X_test = df_hold.iloc[id_recal[i]:id_test_end[i]+1,:].values[:,id_codes_list[i]].copy()
    score_test_list.append(np.mean(X_test, axis=1))
    ws_test_list[i, id_codes_list[i]] = 1./len(id_codes_list[i])
score_test_list = np.concatenate(score_test_list) - 1.

# equally weighted portfolio
np.savez('result/res_russell_EW.npz', 
            score_test_list=score_test_list, 
            ws_test_list=ws_test_list,
            test_date=test_date
        )
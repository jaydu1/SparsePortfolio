import pandas as pd
import numpy as np


########################################################################
#
#  S&P 500
#
########################################################################
n_days_train = 120
n_days_hold = 63

df_close = pd.read_csv('data/sp/sp500_PRC.csv', index_col=[0])
df_open = pd.read_csv('data/sp/sp500_OPENPRC.csv', index_col=[0])
df = df_close/df_open

T, d = df.shape

id_begin = np.where(df.index>=20110101)[0][0]
df_hold = pd.read_csv('data/sp/sp500_RET.csv', index_col=[0]) + 1
id_recal = np.arange(id_begin, len(df.index), n_days_hold)
df_hold.iloc[id_recal,:] = df.iloc[id_recal,:].copy()
df_hold = df_hold.fillna(1.)
n_recal = len(id_recal)

test_date = np.array(df.index[id_begin:])

df_listed = pd.read_csv('data/sp/sp500_listed.csv', index_col=[0])
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

df_rf = pd.read_csv('data/rf.csv')


data = np.load('result/res_sp_EW.npz')
score_test_list, ws_test_list, test_date = data['score_test_list'], data['ws_test_list'], data['test_date']

rf = df_rf.loc[df_rf['date'].isin(test_date), 'rf'].values
ret = score_test_list

cum_ret = np.cumprod(ret+1)
end_ret = cum_ret[-1] - 1
max_drawdown = np.max((np.maximum.accumulate(cum_ret) - cum_ret)/np.maximum.accumulate(cum_ret))
ret = np.log(1+ret)
sharpe_ratio = (np.mean(ret) - np.mean(np.log(1+rf)))/np.std(ret)

print('EW & %.4f & %.4f & %.4f & %d \\\\'%(end_ret, max_drawdown, sharpe_ratio,
                                            int(np.mean(np.sum(ws_test_list>0., axis=1)))))


for i,method in enumerate(['GMV-P', 'GMV-LS', 'GMV-NLS',
                          'MV-P', 'MV-LS', 'MV-NLS']):
    
    data = np.load('result/res_sp_%s.npz'%(method))
    score_test_list, ws_test_list, test_date = data['score_test_list'], data['ws_test_list'], data['test_date']
    
    rf = df_rf.loc[df_rf['date'].isin(test_date), 'rf'].values
    ret = score_test_list# - rf
    
    cum_ret = np.cumprod(ret+1)
    returns = cum_ret[-1] - 1
    risks = np.std(ret)

    max_drawdown = np.max((np.maximum.accumulate(cum_ret) - cum_ret)/np.maximum.accumulate(cum_ret))
    ret = np.log(1+ret)
    sharpe_ratio = (np.mean(ret) - np.mean(np.log(1+rf)))/np.std(ret)
    
    print('%s & %.4f & %.4f & %.4f & %d \\\\'%(method, returns, max_drawdown, sharpe_ratio, 
                                               int(np.mean(np.sum(ws_test_list>0., axis=1)))))



n_days_train =120
n_days_hold = 63

for i,method in enumerate(['LOG', 'EXP']):

    if i == 0:
        a_list = [1.]
        
    else:
        a_list = [5e-2,1e-1,5e-1,1.,1.5]
        
    for a in a_list:
        if i == 0:
            label = method
        else:
            label = method + ' %.2f'%a
        data = np.load('result/res_sp_%s_%.2f.npz'%(method,a))
        score_test_list, ws_test_list, test_date, lambdas = data['score_test_list'], data['ws_test_list'], data['test_date'], data['lambdas']

        rf = df_rf.loc[df_rf['date'].isin(test_date), 'rf'].values
        ret = score_test_list
        
        cum_ret = np.cumprod(ret+1)
        returns = cum_ret[-1] - 1

        max_drawdown = np.max((np.maximum.accumulate(cum_ret) - cum_ret)/np.maximum.accumulate(cum_ret))
        ret = np.log(1+ret)
        sharpe_ratio = (np.mean(ret) - np.mean(np.log(1+rf)))/np.std(ret)
        print('%s-%.2f & %.4f & %.4f & %.4f & %d \\\\'%(method, a, returns, max_drawdown, sharpe_ratio, int(np.mean(np.sum(ws_test_list>0., axis=1)))))
    

# With transaction cost
data = np.load('result/res_sp_EW.npz')
score_test_list, ws_test_list, test_date = data['score_test_list'], data['ws_test_list'], data['test_date']

ws_test_list = np.nan_to_num(ws_test_list, nan=0.0)
for i in range(1,n_recal):
    idx = id_recal[i] - id_begin
    score_test_list[idx-1] = (score_test_list[idx-1] + 1 ) * (
        1 - 1e-3 * np.sum(np.abs(ws_test_list[i,:]-ws_test_list[i-1,:]))
        - 1e-5 * np.sum(np.abs(ws_test_list[i,:]-ws_test_list[i-1,:]) >0.)
    ) -1
    
rf = df_rf.loc[df_rf['date'].isin(test_date), 'rf'].values
ret = score_test_list

cum_ret = np.cumprod(ret+1)
end_ret = cum_ret[-1] - 1
max_drawdown = np.max((np.maximum.accumulate(cum_ret) - cum_ret)/np.maximum.accumulate(cum_ret))
ret = np.log(1+ret)
sharpe_ratio = (np.mean(ret) - np.mean(np.log(1+rf)))/np.std(ret)

print('EW & %.4f & %.4f & %.4f & %d \\\\'%(end_ret, max_drawdown, sharpe_ratio,
                                            int(np.mean(np.sum(ws_test_list>0., axis=1)))))


for i,method in enumerate(['GMV-P', 'GMV-LS', 'GMV-NLS',
                          'MV-P', 'MV-LS', 'MV-NLS']):
    
    data = np.load('result/res_sp_%s.npz'%(method))
    score_test_list, ws_test_list, test_date = data['score_test_list'], data['ws_test_list'], data['test_date']
    
    ws_test_list = np.nan_to_num(ws_test_list, nan=0.0)
    for i in range(1,n_recal):
        idx = id_recal[i] - id_begin
        score_test_list[idx-1] = (score_test_list[idx-1] + 1 ) * (
            1 - 1e-3 * np.sum(np.abs(ws_test_list[i,:]-ws_test_list[i-1,:]))
            - 1e-5 * np.sum(np.abs(ws_test_list[i,:]-ws_test_list[i-1,:]) >0.)
        ) -1
    
    rf = df_rf.loc[df_rf['date'].isin(test_date), 'rf'].values
    ret = score_test_list# - rf
    
    cum_ret = np.cumprod(ret+1)
    returns = cum_ret[-1] - 1
    risks = np.std(ret)

    max_drawdown = np.max((np.maximum.accumulate(cum_ret) - cum_ret)/np.maximum.accumulate(cum_ret))
    ret = np.log(1+ret)
    sharpe_ratio = (np.mean(ret) - np.mean(np.log(1+rf)))/np.std(ret)
    
    print('%s & %.4f & %.4f & %.4f & %d \\\\'%(method, returns, max_drawdown, sharpe_ratio, 
                                               int(np.mean(np.sum(ws_test_list>0., axis=1)))))


for i,method in enumerate(['LOG', 'EXP']):

    if i == 0:
        a_list = [1.]
        
    else:
        a_list = [5e-2,1e-1,5e-1,1.,1.5]
        
    for a in a_list:
        if i == 0:
            label = method
        else:
            label = method + ' %.2f'%a
        data = np.load('result/res_sp_%s_%.2f.npz'%(method,a))
        score_test_list, ws_test_list, test_date, lambdas = data['score_test_list'], data['ws_test_list'], data['test_date'], data['lambdas']
        
        ws_test_list = np.nan_to_num(ws_test_list, nan=0.0)
        for i in range(1,n_recal):
            idx = id_recal[i] - id_begin
            score_test_list[idx-1] = (score_test_list[idx-1] + 1 ) * (
                1 - 1e-3 * np.sum(np.abs(ws_test_list[i,:]-ws_test_list[i-1,:]))
                - 1e-5 * np.sum(np.abs(ws_test_list[i,:]-ws_test_list[i-1,:]) >0.)
            ) -1

        rf = df_rf.loc[df_rf['date'].isin(test_date), 'rf'].values
        ret = score_test_list
        
        cum_ret = np.cumprod(ret+1)
        returns = cum_ret[-1] - 1

        max_drawdown = np.max((np.maximum.accumulate(cum_ret) - cum_ret)/np.maximum.accumulate(cum_ret))
        ret = np.log(1+ret)
        sharpe_ratio = (np.mean(ret) - np.mean(np.log(1+rf)))/np.std(ret)
        print('%s-%.2f & %.4f & %.4f & %.4f & %d \\\\'%(method, a, returns, max_drawdown, sharpe_ratio, int(np.mean(np.sum(ws_test_list>0., axis=1)))))
    



########################################################################
#
#  Russell 2000
#
########################################################################                                               
path_data = './data/russell/'

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

df_rf = pd.read_csv('data/rf.csv')



data = np.load('result/res_russell_EW.npz')
score_test_list, ws_test_list, test_date = data['score_test_list'], data['ws_test_list'], data['test_date']

rf = df_rf.loc[df_rf['date'].isin(test_date), 'rf'].values
ret = score_test_list

cum_ret = np.cumprod(ret+1)
end_ret = cum_ret[-1] - 1
max_drawdown = np.max((np.maximum.accumulate(cum_ret) - cum_ret)/np.maximum.accumulate(cum_ret))
ret = np.log(1+ret)
sharpe_ratio = (np.mean(ret) - np.mean(np.log(1+rf)))/np.std(ret)

print('EW & %.4f & %.4f & %.4f & %d \\\\'%(end_ret, max_drawdown, sharpe_ratio,
                                            int(np.mean(np.sum(ws_test_list>0., axis=1)))))


for i,method in enumerate(['GMV-P', 'GMV-LS', 'GMV-NLS',
                          'MV-P', 'MV-LS', 'MV-NLS']):
    
    data = np.load('result/res_russell_%s.npz'%(method))
    score_test_list, ws_test_list, test_date = data['score_test_list'], data['ws_test_list'], data['test_date']
    
    rf = df_rf.loc[df_rf['date'].isin(test_date), 'rf'].values
    ret = score_test_list
    
    cum_ret = np.cumprod(ret+1)
    returns = cum_ret[-1] - 1
    risks = np.std(ret)

    max_drawdown = np.max((np.maximum.accumulate(cum_ret) - cum_ret)/np.maximum.accumulate(cum_ret))
    ret = np.log(1+ret)
    sharpe_ratio = (np.mean(ret) - np.mean(np.log(1+rf)))/np.std(ret)
    
    print('%s & %.4f & %.4f & %.4f & %d \\\\'%(method, returns, max_drawdown, sharpe_ratio, 
                                               int(np.mean(np.sum(ws_test_list>0., axis=1)))))



for i,method in enumerate(['LOG', 'EXP']):
    if i == 0:
        a_list = [1.]
        
    else:
        a_list = [5e-2,1e-1,5e-1,1.,1.5]
        
    for a in a_list:
        if i == 0:
            label = method
        else:
            label = method + ' %.2f'%a
            
        data = np.load('result/res_russell_%s_%.2f.npz'%(method,a))
        score_test_list, ws_test_list, test_date, lambdas = data['score_test_list'], data['ws_test_list'], data['test_date'], data['lambdas']
        
        rf = df_rf.loc[df_rf['date'].isin(test_date), 'rf'].values
        ret = score_test_list #- rf
        
        cum_ret = np.cumprod(ret+1)
        returns = cum_ret[-1] - 1

        max_drawdown = np.max((np.maximum.accumulate(cum_ret) - cum_ret)/np.maximum.accumulate(cum_ret))
        ret = np.log(1+ret)
        sharpe_ratio = (np.mean(ret) - np.mean(np.log(1+rf)))/np.std(ret)
        print('cv%s-%.2f & %.4f & %.4f & %.4f & %d \\\\'%(method, a, returns, max_drawdown, sharpe_ratio, int(np.mean(np.sum(ws_test_list>0., axis=1)))))
                                                       
for i,method in enumerate(['LOG', 'EXP']):
    if i == 0:
        a_list = [1.]
        
    else:
        a_list = [5e-2,1e-1,5e-1,1.,1.5]
        
    for a in a_list:
        if i == 0:
            label = method
        else:
            label = method + ' %.2f'%a
            
        data = np.load('result/res_russell_%s_%.2f.npz'%(method,a))
        score_test_list, ws_test_list, test_date, lambdas = data['score_test_list'], data['ws_test_list'], data['test_date'], data['lambdas']

        for j in np.arange(n_recal):
            if np.sum(ws_test_list[j,:]>0.)<5:
                score_test_list[id_recal[j]-id_begin:id_test_end[j]-id_begin+1] = 0.
                ws_test_list[j,:] = 0.

        rf = df_rf.loc[df_rf['date'].isin(test_date), 'rf'].values
        ret = score_test_list #- rf
        
        cum_ret = np.cumprod(ret+1)
        returns = cum_ret[-1] - 1

        max_drawdown = np.max((np.maximum.accumulate(cum_ret) - cum_ret)/np.maximum.accumulate(cum_ret))
        ret = np.log(1+ret)
        sharpe_ratio = (np.mean(ret) - np.mean(np.log(1+rf)))/np.std(ret)
        print('no%s-%.2f & %.4f & %.4f & %.4f & %d \\\\'%(method, a, returns, max_drawdown, sharpe_ratio, int(np.mean(np.sum(ws_test_list>0., axis=1)))))
             
             
# With transaction cost
data = np.load('result/res_russell_EW.npz')
score_test_list, ws_test_list, test_date = data['score_test_list'], data['ws_test_list'], data['test_date']


ws_test_list = np.nan_to_num(ws_test_list, nan=0.0)
for i in range(1,n_recal):
    idx = id_recal[i] - id_begin
    score_test_list[idx-1] = (score_test_list[idx-1] + 1 ) * (
        1 - 1e-3 * np.sum(np.abs(ws_test_list[i,:]-ws_test_list[i-1,:]))
        - 1e-5 * np.sum(np.abs(ws_test_list[i,:]-ws_test_list[i-1,:]) >0.)
    ) -1
rf = df_rf.loc[df_rf['date'].isin(test_date), 'rf'].values
ret = score_test_list

cum_ret = np.cumprod(ret+1)
end_ret = cum_ret[-1] - 1
max_drawdown = np.max((np.maximum.accumulate(cum_ret) - cum_ret)/np.maximum.accumulate(cum_ret))
ret = np.log(1+ret)
sharpe_ratio = (np.mean(ret) - np.mean(np.log(1+rf)))/np.std(ret)

print('EW & %.4f & %.4f & %.4f & %d \\\\'%(end_ret, max_drawdown, sharpe_ratio,
                                            int(np.mean(np.sum(ws_test_list>0., axis=1)))))

for i,method in enumerate(['GMV-P', 'GMV-LS', 'GMV-NLS',
                          'MV-P', 'MV-LS', 'MV-NLS']):
    
    data = np.load('result/res_russell_%s.npz'%(method))
    score_test_list, ws_test_list, test_date = data['score_test_list'], data['ws_test_list'], data['test_date']
    
    ws_test_list = np.nan_to_num(ws_test_list, nan=0.0)
    for i in range(1,n_recal):
        idx = id_recal[i] - id_begin
        score_test_list[idx-1] = (score_test_list[idx-1] + 1 ) * (
            1 - 1e-3 * np.sum(np.abs(ws_test_list[i,:]-ws_test_list[i-1,:]))
            - 1e-5 * np.sum(np.abs(ws_test_list[i,:]-ws_test_list[i-1,:]) >0.)
        ) -1
    
    rf = df_rf.loc[df_rf['date'].isin(test_date), 'rf'].values
    ret = score_test_list
    
    cum_ret = np.cumprod(ret+1)
    returns = cum_ret[-1] - 1
    risks = np.std(ret)

    max_drawdown = np.max((np.maximum.accumulate(cum_ret) - cum_ret)/np.maximum.accumulate(cum_ret))
    ret = np.log(1+ret)
    sharpe_ratio = (np.mean(ret) - np.mean(np.log(1+rf)))/np.std(ret)
    
    print('%s & %.4f & %.4f & %.4f & %d \\\\'%(method, returns, max_drawdown, sharpe_ratio, 
                                               int(np.mean(np.sum(ws_test_list>0., axis=1)))))


for i,method in enumerate(['LOG', 'EXP']):
    if i == 0:
        a_list = [1.]
        
    else:
        a_list = [5e-2,1e-1,5e-1,1.,1.5]
        
    for a in a_list:
        if i == 0:
            label = method
        else:
            label = method + ' %.2f'%a
            
        data = np.load('result/res_russell_%s_%.2f.npz'%(method,a))
        score_test_list, ws_test_list, test_date, lambdas = data['score_test_list'], data['ws_test_list'], data['test_date'], data['lambdas']

        ws_test_list = np.nan_to_num(ws_test_list, nan=0.0)
        for i in range(1,n_recal):
            idx = id_recal[i] - id_begin
            score_test_list[idx-1] = (score_test_list[idx-1] + 1 ) * (
                1 - 1e-3 * np.sum(np.abs(ws_test_list[i,:]-ws_test_list[i-1,:]))
                - 1e-5 * np.sum(np.abs(ws_test_list[i,:]-ws_test_list[i-1,:]) >0.)
            ) -1
            
        rf = df_rf.loc[df_rf['date'].isin(test_date), 'rf'].values
        ret = score_test_list #- rf
        
        cum_ret = np.cumprod(ret+1)
        returns = cum_ret[-1] - 1

        max_drawdown = np.max((np.maximum.accumulate(cum_ret) - cum_ret)/np.maximum.accumulate(cum_ret))
        ret = np.log(1+ret)
        sharpe_ratio = (np.mean(ret) - np.mean(np.log(1+rf)))/np.std(ret)
        print('%s-%.2f & %.4f & %.4f & %.4f & %d \\\\'%(method, a, returns, max_drawdown, sharpe_ratio, int(np.mean(np.sum(ws_test_list>0., axis=1)))))
                                                       



for i,method in enumerate(['LOG', 'EXP']):
    if i == 0:
        a_list = [1.]
        
    else:
        a_list = [5e-2,1e-1,5e-1,1.,1.5]
        
    for a in a_list:
        if i == 0:
            label = method
        else:
            label = method + ' %.2f'%a
            
        data = np.load('result/res_russell_%s_%.2f.npz'%(method,a))
        score_test_list, ws_test_list, test_date, lambdas = data['score_test_list'], data['ws_test_list'], data['test_date'], data['lambdas']

        for j in np.arange(n_recal):
            if np.sum(ws_test_list[j,:]>0.)<5:
                score_test_list[id_recal[j]-id_begin:id_test_end[j]-id_begin+1] = 0.
                ws_test_list[j,:] = 0.

        ws_test_list = np.nan_to_num(ws_test_list, nan=0.0)
        for i in range(1,n_recal):
            idx = id_recal[i] - id_begin
            score_test_list[idx-1] = (score_test_list[idx-1] + 1 ) * (
                1 - 1e-3 * np.sum(np.abs(ws_test_list[i,:]-ws_test_list[i-1,:]))
                - 1e-5 * np.sum(np.abs(ws_test_list[i,:]-ws_test_list[i-1,:]) >0.)
            ) -1
                
        rf = df_rf.loc[df_rf['date'].isin(test_date), 'rf'].values
        ret = score_test_list #- rf
        
        cum_ret = np.cumprod(ret+1)
        returns = cum_ret[-1] - 1

        max_drawdown = np.max((np.maximum.accumulate(cum_ret) - cum_ret)/np.maximum.accumulate(cum_ret))
        ret = np.log(1+ret)
        sharpe_ratio = (np.mean(ret) - np.mean(np.log(1+rf)))/np.std(ret)
        print('no%s-%.2f & %.4f & %.4f & %.4f & %d \\\\'%(method, a, returns, max_drawdown, sharpe_ratio, int(np.mean(np.sum(ws_test_list>0., axis=1)))))
    
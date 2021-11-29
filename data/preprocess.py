import os
import sys
import pandas as pd
import numpy as np

'''
NYSE
'''
df = pd.read_csv('data/raw/NYSE.csv')
df = df[(~pd.isna(df['TICKER']))&(~pd.isna(df['RET']))&(pd.isna(df['NAMEENDT']))]
df = df[df['ISSUNO']!=0]
df = df[(df['RET']!='B')&(df['RET']!='C')]
df = df.astype({'RET':float})

df_is = df.groupby('ISSUNO')['TSYMBOL'].apply(lambda x:len(np.unique(x)))
_df = df.drop_duplicates(['TSYMBOL', 'ISSUNO']).reset_index(drop=True)
_df = _df[_df['ISSUNO'].isin(df_is[df_is>1].index)][['TSYMBOL', 'ISSUNO']].reset_index(drop=True)
_df = _df.iloc[_df['ISSUNO'].drop_duplicates(keep='last').index].reset_index(drop=True)

_df = df[['date', 'ISSUNO', 'RET']]
_df = _df.pivot(index='date', columns='ISSUNO', values='RET')
_df = _df[_df.index<=20180101]
_df = _df.loc[:,np.sum(pd.isna(_df))<20]
_df = _df.fillna(0.)

_df.to_csv('data/NYSE/NYSE.csv')



'''
S&P 500
'''
sp500 = pd.read_csv('data/sp500/ticker.csv')
df = pd.read_csv('data/raw/SP500.csv', low_memory=False)
df = df[df['TICKER'].isin(sp500['TICKER_CRDS'])]
df.loc[df['TSYMBOL']=='UAC', 'TICKER'] = 'UAC'
df.loc[df['TSYMBOL']=='NWSA', 'TICKER'] = 'NWSA'
df.loc[df['RET']=='C', 'RET'] = df.loc[df['RET']=='C','PRC'] / df.loc[df['RET']=='C','OPENPRC']
df = df.astype({'PRC':float,'OPENPRC':float,'RET':float})

dict_ticker = {sp500.iloc[i]['TICKER_CRDS']:sp500.iloc[i]['TICKER'] for i in range(len(sp500))}
dict_ticker['UAC'] = 'UAC'
dict_ticker['NWSA'] = 'NWSA'
df.loc[:,'TICKER'] = df['TICKER'].apply(lambda x:dict_ticker[x])

df = df.dropna(subset=['PRC'])
df = df[df['date']>=20100101]

df['value'] = df['PRC']/df['OPENPRC']
_df = df[['date', 'TICKER', 'value']].drop_duplicates()
_df = _df.pivot(index='date', columns='TICKER', values='value')
id_col = (pd.isna(_df).sum(axis=0)==0)

for col in ['PRC','OPENPRC','RET']:
    _df = df[['date', 'TICKER', col]].drop_duplicates()
    _df = _df.pivot(index='date', columns='TICKER', values=col)
    _df = _df.loc[:, id_col]
    _df.to_csv('data/sp500/%s.csv'%col)    



'''
Russell 2000
'''
russell_list = pd.read_csv('data/russel/russell2000_list.txt', header=None).values.flatten()
df = pd.read_csv('data/russel/russell2000.csv')
df = df[df['TSYMBOL'].isin(russell_list)].reset_index(drop=True)
df = df[~pd.isna(df['RET'])].reset_index(drop=True)
df.loc[df['RET']=='C', 'RET'] = df.loc[df['RET']=='C','PRC'] / df.loc[df['RET']=='C','OPENPRC']
for col in ['PRC','OPENPRC','RET']:
    _df = df[['date', 'TSYMBOL', col]].drop_duplicates()
    _df = _df.pivot(index='date', columns='TSYMBOL', values=col)
    _df= _df.astype(float)
    _df.to_csv('data/russell/%s_russell.csv'%col)
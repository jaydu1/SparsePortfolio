import os
import sys
import pandas as pd
import numpy as np


path_data = './data/'

########################################################################
#
#  NYSE
#
########################################################################
df = pd.read_csv(path_data+'raw/NYSE.csv')
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

_df.to_csv(path_data+'NYSE/NYSE.csv')



########################################################################
#
#  S&P 500
#
########################################################################
'''
Process data from CRSP
'''
sp500 = pd.read_csv(path_data+'raw/ticker_sp.csv')

df = pd.read_csv(path_data+'raw/sp500.csv', low_memory=False)
df = df[df['TICKER'].isin(sp500['TICKER_CRDS'])]

df = df[df['PERMNO']!=92389]
df = df[~((df['PERMNO']==75631)&(df['TICKER']=='TFC'))]

df.loc[df['TSYMBOL']=='NWSA', 'TICKER'] = 'NWSA'
df.loc[df['RET']=='C', 'RET'] = df.loc[df['RET']=='C','PRC'] / df.loc[df['RET']=='C','OPENPRC']

# merge
df.loc[df['TICKER']=='BBT', 'TICKER'] = 'TFC'
df.loc[df['TICKER']=='MYL', 'TICKER'] = 'VTRS'

# acquired
df.loc[
    (df['TICKER']=='LKQ') &
    (df['date']>df.loc[df['TICKER']=='ARG', 'date'].max()), 'TICKER'
] = 'ARG'
df = df[df['TICKER']!='LKQ']
df.loc[df['TICKER']=='CBS', 'TICKER'] = 'VIAC'

# splitted
df.loc[df['TICKER']=='SYMC', 'TICKER'] = 'NLOK'
df.loc[df['TICKER']=='GL', 'TICKER'] = 'TMK'
df.loc[
    (df['TICKER']=='ARNC') &
    (df['date']<df.loc[df['TICKER']=='HWM', 'date'].min()), 'TICKER'
] = 'HWM'
df = df[df['TICKER']!='ARNC']
df.loc[
    (df['TSYMBOL']=='UA') &
    (df['date']<=df.loc[df['TSYMBOL']=='UAC', 'date'].max()), 'TICKER'
] = 'UAA'
df.loc[df['TICKER']=='CCEP', 'TICKER'] = 'CCE'
df = df[df['TICKER']!='FCPT']
df = df[df['TICKER']!='VIA']

df = df.astype({'PRC':float,'OPENPRC':float,'RET':float})

dict_ticker = {sp500.iloc[i]['TICKER_CRDS']:sp500.iloc[i]['TICKER'] for i in range(len(sp500))}
dict_ticker['UAC'] = 'UA'
dict_ticker['NWSA'] = 'NWSA'
df.loc[:,'TICKER'] = df['TICKER'].apply(lambda x:dict_ticker[x])

print(len(df['TICKER'].unique()))
df = df.dropna(subset=['PRC'])
print(len(df['TICKER'].unique()))

df = df[df['date']>=20100101]

for col in ['PRC','OPENPRC','RET']:
    _df = df[['date', 'TICKER', col]].drop_duplicates()
    _df = _df.pivot(index='date', columns='TICKER', values=col)
    _df.to_csv(path_data+'sp/sp500_%s.csv'%col)

print(_df.shape)

df_sp500_history = pd.read_csv(path_data+'raw/sp500_history.csv')
sp500 = df_sp500_history[df_sp500_history['date']<'2021-01-01'].iloc[-1]['tickers'].split(',')

df_sp500_history = df_sp500_history[(df_sp500_history['date']>='2010-01-01')&
                                    (df_sp500_history['date']<'2021-01-01')].reset_index(drop=True)
df_sp500_history['tickers'] = df_sp500_history['tickers'].apply(
    lambda x : x.split(',')
)
replace_tuple = [
    ('ARNC','HWM'),
    ('LKQ','ARG'),
    ('GL','TMK'),
    ('SYMC','NLOK'),
    ('CBS','VIAC'),
    ('MYL','VTRS')
]
for t in replace_tuple:
    df_sp500_history['tickers'] = df_sp500_history['tickers'].apply(
        lambda x : [t[1] if x==t[0] else i for i in x]
    )
tickers = np.array(_df.columns)
xsorted = np.argsort(tickers)
n_days = df_sp500_history.shape[0]
res = np.zeros((n_days, len(tickers)))
for i in range(n_days):
    indices = [np.where(tickers==code)[0][0] for code in df_sp500_history.iloc[i,1] if code in tickers]
    res[i, indices] = 1

df_listed = pd.DataFrame(res, columns=_df.columns, index=df_sp500_history['date'])
df_listed.to_csv(path_data+'sp/sp500_listed.csv')


########################################################################
#
#  Russell 2000
#
########################################################################
'''
Generate index list
'''
russell2000 = pd.read_csv(path_data+'raw/ticker_russell.csv')
tickers = np.unique(russell2000['TICKER'])

df_listed = pd.read_excel(path_data+'raw/Russell 2000 Historical Components 1996 - 2021.xlsx', 
                          header=1, index_col=0, dtype={'Ticker':str})
ticker = np.array(df_listed.index)
ticker[ticker==True] = 'TRUE'
df_listed.index = ticker
df_listed = df_listed.drop(['Company Name','ISIN Code','Sector'], axis=1)
df_listed = df_listed.transpose()
df_listed.index = pd.to_datetime(df_listed.index).strftime('%Y%m%d')

df_listed = df_listed[(df_listed.index>='20040101')&(df_listed.index<'20210101')].sort_index()
df_listed = df_listed[df_listed.columns[~df_listed.isnull().all()]]

codes = ['OPEN (Solu)','OPEN (Ope)']
names = np.array(df_listed.columns)
for i,name in enumerate(df_listed.columns):
    if ' ' in name:
        if name in tickers or name in codes:
            continue
        if name.split(' ')[0] not in df_listed.columns:
            names[i] = name.split(' ')[0]
df_listed.columns = names

dict_ticker = {
    'AABA':'YHOO',
    #'BBT':'TFC',
 'BHGE':'BKR',
 'BTUUQ':'BTU',
 'COG':'CTRA',
 'CTL':'LUMN',
 'CVC':'RMG',
 'DWDP':'DD (Old)',
 'EKDKQ':'KODK',
 'ESV':'VAL',
 'FII':'FHI',
 'HCP':'PEAK',
 'HRS':'LHX',
 'JEC':'J',
 'KDP':'DOC (Doc)',
 'KORS':'CPRI',
 'KRFT':'KRA (Kra)',
 'LB':'BBWI',
 'MWV':'WRK',
 'RSHCQ':'RSH',
 'RX':'IMS',
 'SYMC':'NLOK',
 'TMK':'GL',
 'UAA':'UA.A',
               
 'ALTR':'ALTR (Alt)', 
 'BEAM':'BEAM (Sun)', 
 'IGT':'IGT (Old)', 
 'LIFE':'LIFE (Tech)', 
 'LSI':'LSI (LSI)', 
 'MMI':'MMI (Mot)', 
 'NSM':'NSM (Nat)', 
 'NVLS':'NVLS (Nove)', 
 'Q':'Q (Qwe)', 
 'SUN':'SUN (Sun)',
    
 'GAS':'GAS (Nico)'
    
}

for t_new in dict_ticker:
    t_old = dict_ticker[t_new]
    
    if t_old in df_listed.columns and t_new in df_listed.columns:
        df_listed.loc[pd.isna(df_listed[t_old]), t_old] = df_listed.loc[pd.isna(df_listed[t_old]), t_new]
        df_listed = df_listed.drop(t_new, axis=1)
    else:
        col_names = np.array(list(df_listed.columns))
        col_names[col_names==t_old] = t_new
        df_listed.columns = col_names

df_listed = df_listed.replace('X', 1)
df_listed = df_listed.replace(np.nan, 0)

df_listed = df_listed[df_listed.columns[~(df_listed.sum()==0)]]

# remove stock issued outside of US
df_listed = df_listed.drop('XPRO', axis=1)


'''
Process data from CRSP
'''
df = pd.read_csv(path_data+'raw/russell2000.csv', low_memory=False)

df = df[df['date']>=20040101]
df = df[df['PERMNO']!=92389]
df = df[~((df['PERMNO']==75631)&(df['TICKER']=='TFC'))]
df = df[~pd.isna(df[['PRC', 'RET', 'OPENPRC']]).all(axis=1)]

# acquired
_df = df[(df['TICKER']=='BUSE') & (df['date']>df[df['TICKER']=='FCFP']['date'].max())].copy()
_df[['PERMNO', 'TICKER', 'TSYMBOL']] = df[df['TICKER']=='FCFP'][['PERMNO', 'TICKER', 'TSYMBOL']].iloc[0,:]
df = df.append(_df)

_df = df[(df['TICKER']=='TSYS') & (df['date']>df[df['TICKER']=='NTSP']['date'].max())].copy()
_df[['PERMNO', 'TICKER', 'TSYMBOL']] = df[df['TICKER']=='NTSP'][['PERMNO', 'TICKER', 'TSYMBOL']].iloc[0,:]
df = df.append(_df)
df = df.reset_index(drop=True)

# df.loc[df['TSYMBOL']=='UAC', 'TICKER'] = 'UAC'
df.loc[df['TSYMBOL']=='NWSA', 'TICKER'] = 'NWSA'
df.loc[df['RET']=='C', 'RET'] = df.loc[df['RET']=='C','PRC'] / df.loc[df['RET']=='C','OPENPRC']

# merge
df.loc[df['TICKER']=='BBT', 'TICKER'] = 'TFC'
df.loc[df['TICKER']=='MYL', 'TICKER'] = 'VTRS'

# acquired
df.loc[
    (df['TICKER']=='LKQ') &
    (df['date']>df.loc[df['TICKER']=='ARG', 'date'].max()), 'TICKER'
] = 'ARG'
df = df[df['TICKER']!='LKQ']
df.loc[df['TICKER']=='CBS', 'TICKER'] = 'VIAC'

# splitted
df.loc[df['TICKER']=='SYMC', 'TICKER'] = 'NLOK'
df.loc[df['TICKER']=='GL', 'TICKER'] = 'TMK'
df.loc[
    (df['TICKER']=='ARNC') &
    (df['date']<df.loc[df['TICKER']=='HWM', 'date'].min()), 'TICKER'
] = 'HWM'
df = df[df['TICKER']!='ARNC']
df.loc[
    (df['TSYMBOL']=='UA') &
    (df['date']<=df.loc[df['TSYMBOL']=='UAC', 'date'].max()), 'TICKER'
] = 'UAA'
df.loc[df['TICKER']=='CCEP', 'TICKER'] = 'CCE'
df = df[df['TICKER']!='FCPT']
df = df[df['TICKER']!='VIA']
df = df[df['TSYMBOL'].isin(russell2000['TICKER_CRDS']) | df['TICKER'].isin(russell2000['TICKER_CRDS'])]
df = df[df['PERMNO'].isin(russell2000['PERMNO'])]

dict_ticker = {(russell2000.iloc[i]['PERMNO'],russell2000.iloc[i]['TICKER_CRDS']):
               russell2000.iloc[i]['TICKER'] for i in range(len(russell2000))}
id_t = ~df['TSYMBOL'].isin(dict_ticker.values())

df = df.loc[~(id_t & 
              (~df.loc[:, ['PERMNO','TICKER']].apply(
                  lambda x:(x['PERMNO'],x['TICKER']) in dict_ticker.keys(), axis=1)))]
df.loc[id_t, 'TSYMBOL'] = df.loc[id_t, ['PERMNO','TICKER']].apply(
    lambda x:dict_ticker[(x['PERMNO'],x['TICKER'])], axis=1)

df.loc[df['RET']=='C', 'RET'] = df.loc[df['RET']=='C','PRC'] / df.loc[df['RET']=='C','OPENPRC']
df = df.astype({'PRC':float,'OPENPRC':float,'RET':float})
df = df.dropna(subset=['PRC'])

df = df[~((df['PERMNO']==90365) & (df['date']<=20040924))]

# not in Russell2000 during the period
df = df[df['TICKER']!='AVN']

for col in ['PRC','OPENPRC','RET']:
    _df = df[['date', 'TSYMBOL', col]].drop_duplicates()
    _df = _df.pivot(index='date', columns='TSYMBOL', values=col)
    _df= _df.astype(float)
    _df.to_csv(path_data+'russell/russell2000_%s.csv'%col)    
    print(_df.shape)

df_listed = df_listed[_df.columns]
df_listed.to_csv(path_data+'russell/russell2000_listed.csv')    
#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:

os.makedirs('./img', exist_ok=True)
style = 'seaborn'
palette = sns.color_palette('deep')


# # Data Inspection

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
df_sp500 = df_hold.copy()


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
df_hold.iloc[id_recal,:] = df.iloc[id_recal,:].copy()
df_russell2000 = df_hold.copy()


corr_sp500 = df_sp500.corr(method='pearson')
corr_russell2000 = df_russell2000.corr(method='pearson')



fig, axes = plt.subplots(1, 2, figsize=(23,10))
# fig.tight_layout()
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.95, 0.15, 0.02, 0.7])

sns.heatmap(corr_sp500, cmap='RdYlGn', vmax=1.0, vmin=-0.5, ax=axes[0],
                cbar=True, cbar_ax=cbar_ax)

sns.heatmap(corr_russell2000, cmap='RdYlGn', vmax=1.0, vmin=-0.5, ax=axes[1],
                 cbar=False, cbar_ax=None)

# plt.yticks(rotation=0) 
# plt.xticks(rotation=90) 
axes[0].set_xticks([])
axes[1].set_xticks([])
axes[0].set_yticks([])
axes[1].set_yticks([])
# axes[0].set_xlabel('(a)', fontsize=24)
# axes[1].set_xlabel('(b)', fontsize=24)

cbar = axes[0].collections[0].colorbar
cbar.ax.tick_params(labelsize=20)

plt.savefig('img/cor.png', dpi=1200, bbox_inches='tight', pad_inches=0)
plt.show()


style = 'seaborn'
palette = sns.color_palette('deep')
plt.style.use(style)
sns.set(font_scale = 2)

_df = pd.DataFrame(df_sp500.std())
_df['Data'] = 'S&P 500'

_df2 = pd.DataFrame(df_russell2000.std())
_df2['Data'] = 'Russell 2000'

df_std = pd.concat([_df,_df2]).reset_index(drop=True)
df_std.columns = ['Standard Deviation', 'Data']

fig, ax = plt.subplots(1, 1, figsize=(23,10))
sns.histplot(df_std, stat='proportion', kde=False, ax=ax, 
             x='Standard Deviation', hue='Data', common_norm=False)
ax.set_ylabel('Empirical Proportion')
ax.patch.set_alpha(0.5)
# ax.set_xlabel('(c)')
ax.set_xlim([-0.0, 0.3])

plt.savefig('img/std.png', dpi=1200, bbox_inches='tight', pad_inches=0)
sns.set(font_scale = 1)


# # Single Crossing


import scipy.stats as ss


plt.style.use(style)

x = np.linspace(-5, 5, 50000)

fig, axes = plt.subplots(1,3,figsize=(20,5), sharey=True)
fig.tight_layout()

y_cdf_1 = ss.norm.cdf(x, 0, 1.0)
y_cdf_2 = ss.norm.cdf(x, 1.0, 1.0)
axes[0].plot(x, y_cdf_1, label='$Z_1$', color=palette[0])
axes[0].plot(x, y_cdf_2, label='$Z_2$', color=palette[2])
axes[0].legend(fontsize=20)
# axes[0].set_xlabel('(a)', fontsize=24)

y_cdf_1 = ss.norm.cdf(x, 0, 1.0)
y_cdf_2 = ss.norm.cdf(x, 1.0, 2.)
axes[1].plot(x, y_cdf_1, label='$Z_1$', color=palette[0])
axes[1].plot(x, y_cdf_2, label='$Z_2$', color=palette[2])
axes[1].legend(fontsize=24)
# axes[1].set_xlabel('(b)', fontsize=24)

y_cdf_1 = ss.norm.cdf(x, 0, 2.)
y_cdf_2 = ss.norm.cdf(x, 1.0, 1.0)
axes[2].plot(x, y_cdf_1, label='$Z_1$', color=palette[0])
axes[2].plot(x, y_cdf_2, label='$Z_2$', color=palette[2])
axes[2].legend(fontsize=24)
# axes[2].set_xlabel('(c)', fontsize=24)

for i in range(3):
    axes[i].tick_params(axis='x', labelsize=20)
    axes[i].tick_params(axis='y', labelsize=20)
    axes[i].patch.set_alpha(0.5)

plt.savefig('img/single_crossing.png', dpi=1200, bbox_inches='tight', pad_inches=0.05)


# # NYSE

df = pd.read_csv('data/nyse/NYSE.csv', index_col=[0])
X = df.values
n, d = X.shape
X += 1.

data = np.load('result/res_mv_efficient_frontier.npz')
ws_list = data['ws_list']
ret = np.dot(ws_list, X.T) - 1.
cum_ret = np.cumprod(ret+1, axis=-1)
returns = cum_ret[:,-1] - 1
risks = np.std(ret, axis=-1)
n_ws = np.sum(ws_list>0., axis=1)

plt.style.use(style)
fig, axes = plt.subplots(1,2, figsize=(15,4), sharey=True)
# fig.tight_layout()

ax1 = axes[0]
ax2 = ax1.twinx()

df_res = pd.DataFrame()
_df = pd.DataFrame()
_df['risk'] = risks[n_ws<3500]
_df['return'] = returns[n_ws<3500]
_df['s'] = n_ws[n_ws<3500]
df_res = df_res.append(_df)

ax1 = sns.regplot(x="s", 
           y="return", order=3,
           data=df_res, ax=ax1, scatter_kws={'alpha':0.2, 'linewidth':0.1})
ax2 = sns.regplot(x="s", 
           y="risk", order=3, color=palette[2],
           data=df_res, ax=ax2, scatter_kws={'alpha':0.2, 'linewidth':0.1})
ax1.set_xlim([0,25])
ax1.set_title('Mean-Variance Portfolios', fontsize=22)
ax1.set_xlabel('number of assets', fontsize=20)
ax1.set_ylabel('return', color=palette[0], fontsize=20)
ax1.set_ylim([0,125])
ax2.set_ylim([0., 1.5])
ax2.axes.yaxis.set_visible(False)
ax1.patch.set_alpha(0.5)
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)

data = np.load('result/res_spo_efficient_frontier_LOG.npz')
ws_list = data['ws_list'][0,:,:]
ret = np.dot(X, ws_list) - 1.
cum_ret = np.cumprod(ret+1, axis=0)
returns = cum_ret[-1,:] - 1
risks = np.std(ret, axis=0)
n_ws = np.sum(ws_list>0., axis=0)

ax1 = axes[1]
ax2 = ax1.twinx()

df_res = pd.DataFrame()
_df = pd.DataFrame()
_df['risk'] = risks[n_ws<=28]
_df['return'] = returns[n_ws<=28]
_df['s'] = n_ws[n_ws<=28]
df_res = df_res.append(_df)

ax1 = sns.regplot(x="s", 
           y="return", order=3,
           data=df_res, ax=ax1, scatter_kws={'alpha':0.2, 'linewidth':0.1})
ax2 = sns.regplot(x="s", 
           y="risk", order=3, color=palette[2],
           data=df_res, ax=ax2, scatter_kws={'alpha':0.2, 'linewidth':0.1})
ax1.set_xlim([0,25])
ax1.set_title('Log-Optimal Portfolios', fontsize=22)
ax1.set_xlabel('number of assets', fontsize=20)
ax1.axes.yaxis.set_visible(False)
ax2.set_ylabel('risk', color=palette[2], fontsize=20)
ax2.yaxis.set_ticks(np.arange(0.1, 1.4, 0.3))
ax2.set_ylim([0., 1.5])
ax1.tick_params(axis='x', labelsize=14)
ax2.tick_params(axis='y', labelsize=14)

ax1.patch.set_alpha(0.5)

plt.savefig('img/simulation_frontier.png', dpi=1200, bbox_inches='tight', pad_inches=0)


plt.style.use('default')

a = 1.
fig, axes = plt.subplots(1,2,
                         figsize=(16,4))
for i,method in enumerate(['LOG', 'EXP']):

    if i == 0:
        label = method
    else:
        label = method + '-%.2f'%a
    data = np.load('result/res_ex1_%s_%.2f.npz'%(method,a))

    ws, lambdas, gaps, n_iters, n_actives = data['ws'], data['lambdas'], data['gaps'], data['n_iters'], data['n_actives']


    mat = n_actives[:, :int(1e5)]/np.max(n_actives)
    if i==1:
        mat = mat[:int(np.sum(lambdas>=1e-1)),:]
    # plt.xscale('log')
    pcm = axes[i].imshow(
        mat, aspect='auto', extent=(1, mat.shape[1]+1, mat.shape[0]+1, 1),
         cmap=plt.get_cmap('Blues')
    )

    ny = mat.shape[0]
    no_labels = 4 if i==0 else 3
    y_positions = np.linspace(0,ny,no_labels)
    axes[i].set_yticks(y_positions)
    if i==1:
        axes[i].set_yticklabels(np.flip(np.linspace(-2,0,no_labels)).astype(int))
    else:
        axes[i].set_yticklabels(np.flip(np.linspace(-3,0,no_labels)).astype(int))

    axes[i].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    axes[i].set_xlabel('Iteration', fontsize=18)
    axes[i].set_title('Screening Ratio (%s)'%label, fontsize=20)
    
    axes[i].tick_params(axis='x', labelsize=14)
    axes[i].tick_params(axis='y', labelsize=14)
    
axes[0].set_ylabel('$\log_10(\lambda/\lambda_{\max})$', fontsize=18)
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.93, 0.15, 0.03, 0.7])
cbar = fig.colorbar(pcm, cax=cbar_ax)
cbar.ax.tick_params(labelsize=18)

plt.savefig('img/simulation_screen_ratio.png', dpi=1200, bbox_inches='tight', pad_inches=0)




plt.style.use(style)
fig, axes = plt.subplots(1,3,figsize=(16,4))
labels = ['LOG', 'EXP-1.00']
a = 1.00
colors = ['tab:blue', 'tab:orange']
for i,func_name in enumerate(['LOG', 'EXP']):
    label = labels[i]
    data = np.load('result/res_ex2_%s_%.2f.npz'%(func_name,a))
    res = data['res']
    lambdas = data['lambdas']
    
    df_res = pd.DataFrame.from_dict({"log_lambda_ratio":np.log10(lambdas[1:]/lambdas[0]), "value":res})

    ax = sns.regplot(x="log_lambda_ratio", y="value", data=df_res, 
                ax=axes[i], color=colors[i],
                lowess=False, scatter_kws={'s':2})
    ax.set_title('Relative Execution Time (%s)'%label, fontsize=17)
    ax.set_ylabel('Ratio of screened to unscreened time', fontsize=15)
    ax.set_xlabel('$\log_{10}(\lambda/\lambda_{\max})$', fontsize=15)
    ax.set_ylim([0,1])
    
    if i==0:
        n = 3
        x_positions = np.linspace(-n+1,0,n)
        axes[i].set_xticks(x_positions)
        axes[i].set_xticklabels(np.linspace(-n+1,0,n).astype(int))
    else:
        n = 2
        x_positions = np.linspace(-n+1,0,n+1)
        axes[i].set_xticks(x_positions)
        axes[i].set_xticklabels(['-1','0.5','0'])
        
axes[1].set_ylabel('')

df_res = pd.DataFrame()
lam = 5e-1
for i,func_name in enumerate(['LOG', 'EXP']):
    label = labels[i]
    data = np.load('result/res_ex3_%s_%.2f.npz'%(func_name, lam))
    tols, time_screened, time_unscreened = data['tols'], data['time_screened'], data['time_unscreened'] 
    
    _df_res = pd.DataFrame()
    _df_res['Duality Gap'] = tols
    _df_res['Time'] = time_screened
    _df_res['Model'] = label
    _df_res['Screening'] = 'screened'
    df_res = df_res.append(_df_res)
    
    _df_res = pd.DataFrame()
    _df_res['Duality Gap'] = tols
    _df_res['Time'] = time_unscreened
    _df_res['Model'] = label
    _df_res['Screening'] = 'unscreened'
    df_res = df_res.append(_df_res)
    
df_res = df_res.reset_index(drop=True)
sns.lineplot(x="Time", y="Duality Gap",
             hue="Model", style="Screening",
             data=df_res, ax=axes[2])
axes[2].legend(frameon=True, shadow=False, 
                 framealpha=0.6, columnspacing=1)
axes[2].set_yscale('log')
axes[2].set_ylabel('Duality Gap', fontsize=15)
axes[2].set_xlabel('Time/s', fontsize=15)
axes[2].set_title('Convergence Rate ($\lambda/\lambda_\max=5e-1$)', fontsize=17)


for i in range(3):
    axes[i].patch.set_alpha(0.5)
    axes[i].tick_params(axis='x', labelsize=12)
    axes[i].tick_params(axis='y', labelsize=12)
    
plt.savefig('img/simulation_screen_time.png', dpi=1200, bbox_inches='tight', pad_inches=0)


# # SP500

import empyrical as ep
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import scipy as sp

from matplotlib.ticker import FuncFormatter
def percentage(x, pos):
    """
    Adds percentage sign to plot ticks.
    """

    return '%.0f%%' % x
def plot_annual_returns(returns, ax=None, **kwargs):
    """
    Plots a bar graph of returns by year.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to plotting function.
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    x_axis_formatter = FuncFormatter(percentage)
    ax.xaxis.set_major_formatter(FuncFormatter(x_axis_formatter))
    ax.tick_params(axis='x', which='major')

    ann_ret_df = pd.DataFrame(
        ep.aggregate_returns(
            returns,
            'yearly'))

    ax.axvline(
        100 *
        ann_ret_df.values.mean(),
        color='steelblue',
        linestyle='--',
        lw=4,
        alpha=0.7)
    (100 * ann_ret_df.sort_index(ascending=False)
     ).plot(ax=ax, kind='barh', alpha=0.70, **kwargs)
    ax.axvline(0.0, color='black', linestyle='-', lw=3)

    ax.set_ylabel('Year')
    ax.set_xlabel('Returns')
    ax.set_title("Annual returns")
    ax.legend(['Mean'], frameon=True, framealpha=0.5)
    return ax

def plot_monthly_returns_heatmap(returns, ax=None, **kwargs):
    """
    Plots a heatmap of returns by month.
    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy, noncumulative.
         - See full explanation in tears.create_full_tear_sheet.
    ax : matplotlib.Axes, optional
        Axes upon which to plot.
    **kwargs, optional
        Passed to seaborn plotting function.
    Returns
    -------
    ax : matplotlib.Axes
        The axes that were plotted on.
    """

    if ax is None:
        ax = plt.gca()

    monthly_ret_table = ep.aggregate_returns(returns, 'monthly')
    monthly_ret_table = monthly_ret_table.unstack().round(3)

    sns.heatmap(
        monthly_ret_table.fillna(0) *
        100.0,
        annot=True,
        annot_kws={"size": 9},
        alpha=1.0,
        center=0.0,
        cbar=False,
        cmap=matplotlib.cm.RdYlGn,
        ax=ax, **kwargs)
    ax.set_ylabel('Year')
    ax.set_xlabel('Month')
    ax.set_title("Monthly returns (%)")
    return ax


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


returns_list = []
positions_list = []

def get_r_p(score_test_list, ws_test_list):
    returns = pd.Series(score_test_list, index=pd.to_datetime(test_date.astype(str)), name='returns')
    positions = pd.DataFrame(ws_test_list, index=df.index[id_recal], columns=df.columns)
    positions = positions.merge(df.iloc[id_begin:,0:1].rename({'A':'cash'}, axis=1), how='outer', left_index=True,right_index=True)
    positions = positions.fillna(method="ffill")
    positions.index = pd.to_datetime(positions.index.astype(str))
    positions = positions * 1000
    positions['cash'] = 0.
    return returns, positions

data = np.load('result/res_sp_%s.npz'%('MV-NLS'))
score_test_list, ws_test_list, test_date = data['score_test_list'], data['ws_test_list'], data['test_date']

returns, positions = get_r_p(score_test_list, ws_test_list)
returns_list.append(returns)
positions_list = [positions]

a = 1.
for method in ['LOG', 'EXP']:
    data = np.load('result/res_sp_%s_%.2f.npz'%(method,a))
    score_test_list, ws_test_list, test_date, lambdas = data['score_test_list'], data['ws_test_list'], data['test_date'], data['lambdas']
    returns, positions = get_r_p(score_test_list, ws_test_list)
    returns_list.append(returns)
    positions_list = [positions] 



plt.style.use(style)
plt.tight_layout()
fig, axes = plt.subplots(1,3, figsize=(12,3), sharey=True)
methods = ['MV-NLS', 'LOG', "EXP-1.00"]
for j,returns in enumerate(returns_list):
    
    ax = plot_annual_returns(returns, ax=axes[j])
    ax.set_xlabel('')
    ax.set_title(methods[j], fontsize=16)
    ax.get_legend().remove()
    ax.set_xlim(0,100)

fig.subplots_adjust(left=0.1)  
axes[0].text(-25.0, 5.0, "Annual Returns", size=16, rotation=90.,
         ha="center", va="center", weight='bold'
         )    
axes[0].patch.set_alpha(0.5)
axes[1].patch.set_alpha(0.5)
axes[2].patch.set_alpha(0.5)
plt.savefig('img/sp500_returns_1.png', dpi=1200, bbox_inches='tight', pad_inches=0)

plt.tight_layout()
fig, axes = plt.subplots(1,3, figsize=(12,3), sharey=True)
methods = ['MV-NLS', 'LOG', "EXP-1.00"]
for j,returns in enumerate(returns_list):
    ax = plot_monthly_returns_heatmap(returns, ax=axes[j])
    ax.set_title('', fontsize=20)
    if j>0:
        ax.set_ylabel('')
fig.subplots_adjust(left=0.1, right=0.95)      
axes[0].text(-3.0, 5.0, "Monthly Returns", size=16, rotation=90.,
         ha="center", va="center", weight='bold'
         )   
plt.savefig('img/sp500_returns_2.png', dpi=1200, bbox_inches='tight', pad_inches=0)



plt.style.use(style)
plt.tight_layout()
fig, axes = plt.subplots(1,3, figsize=(12,3), sharey=True)
methods = ['MV-NLS', 'LOG', "EXP-1.00"]
for j,returns in enumerate(returns_list):
    
    ax = plot_annual_returns(returns, ax=axes[j])
    ax.set_xlabel('')
    ax.set_title(methods[j], fontsize=16)
    ax.get_legend().remove()
    ax.set_xlim(0,100)

fig.subplots_adjust(left=0.1)  
axes[0].text(-25.0, 5.0, "Annual Returns", size=16, rotation=90.,
         ha="center", va="center", weight='bold'
         )    
axes[0].patch.set_alpha(0.5)
axes[1].patch.set_alpha(0.5)
axes[2].patch.set_alpha(0.5)
plt.savefig('img/sp500_returns_1.png', dpi=1200, bbox_inches='tight', pad_inches=0)

plt.tight_layout()
fig, axes = plt.subplots(1,3, figsize=(12,3), sharey=True)
methods = ['MV-NLS', 'LOG', "EXP-1.00"]
for j,returns in enumerate(returns_list):
    ax = plot_monthly_returns_heatmap(returns, ax=axes[j])
    ax.set_title('', fontsize=20)
    if j>0:
        ax.set_ylabel('')
fig.subplots_adjust(left=0.1, right=0.95)      
axes[0].text(-3.0, 5.0, "Monthly Returns", size=16, rotation=90.,
         ha="center", va="center", weight='bold'
         )   
plt.savefig('img/sp500_returns_2.png', dpi=1200, bbox_inches='tight', pad_inches=0)



import matplotlib.pyplot as plt
import seaborn as sns

n_days_train =120
n_days_hold = 63
plt.style.use(style)
fig, axes = plt.subplots(2,2, figsize=(15,7), sharex=True)
n_ws = np.arange(1,51)
top_n = 40
ls = ['-','--','-.',':', '--']
colors = sns.color_palette("Set2")
icolor = 0
izorder = 0
for i,method in enumerate(['LOG', 'EXP']):
    
    if i == 0:
        a_list = [1.]
        
    else:
        a_list = [5e-2,1e-1,5e-1,1.,1.5]
        
    for ia, a in enumerate(a_list):
        if i == 0:
            label = method
        else:
            label = method + ' %.2f'%a
        data = np.load('result/res_sp_%s_%.2f_n.npz'%(method,a))
        score_test_list, ws_test_list, test_date, n_ws = data['score_test_list'], data['ws_test_list'], data['test_date'], data['n_ws']
        ret = score_test_list #- rf
                
        cum_ret = np.nancumprod(score_test_list+1,axis=0)
        returns = cum_ret[-1, :] - 1
        risks = np.nanstd(score_test_list, axis=0)

        rf = df_rf.loc[df_rf['date'].isin(test_date), 'rf'].values.reshape((-1,1))

        max_drawdown = np.max((np.maximum.accumulate(cum_ret, axis=0) - cum_ret)/np.maximum.accumulate(cum_ret, axis=0), axis=0)
        
        sharpe_ratio = np.nanmean(np.log(1+score_test_list) - np.log(1+rf), axis=0)/np.nanstd(np.log(1+score_test_list), axis=0)

        n_ws_ = np.mean(np.sum(ws_test_list>0, axis=1), axis=0)

        i_sr = np.argmax(sharpe_ratio)
        
    
        axes[0,0].plot(n_ws[:top_n], returns[:top_n], ls[ia], marker='^', 
                       c=colors[icolor], label=label, zorder=6-izorder)
        axes[0,1].plot(n_ws[:top_n], max_drawdown[:top_n], ls[ia], marker='^', 
                       c=colors[icolor], label=label, zorder=6-izorder)
        axes[1,0].plot(n_ws[:top_n], sharpe_ratio[:top_n], ls[ia], marker='^', 
                       c=colors[icolor], label=label, zorder=6-izorder)
        axes[1,1].plot(n_ws[:top_n], n_ws_[:top_n], ls[ia], marker='^', 
                       c=colors[icolor], label=label, zorder=6-izorder)
        icolor += 1
        izorder += 1
        
titles = ['Return', 'Max Drawdown', 'Sharpe Ratio', 'Average Number of Assets'] 
for i in range(2):
    for j in range(2):
        axes[i,j].set_title(titles[2*i+j], fontsize=20)
        
        if i==1:
            axes[i,j].set_xlabel('$\|\|w\|\|_0$', fontsize=16)
plt.tight_layout()
fig.subplots_adjust(right=0.9)
axes[1,1].legend(bbox_to_anchor=(1.25, 1.3), loc='upper right', ncol=1, frameon=True, shadow=False, 
                 framealpha=0.3, columnspacing=4, title='Model', fontsize=12)

for i in range(2):
    for j in range(2):
        axes[i,j].patch.set_alpha(0.5)
        axes[i,j].tick_params(axis='x', labelsize=12)
        axes[i,j].tick_params(axis='y', labelsize=12)
plt.savefig('img/sp500_n.png', dpi=1200, bbox_inches='tight', pad_inches=0)




# # Russell 2000
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

plt.style.use(style)
fig, axes = plt.subplots(2,2, figsize=(15,7), sharex=True)
top_n = 500
n_ws = np.arange(5,top_n,20)

ls = ['-','--','-.',':', '--']
colors = sns.color_palette("Set2")
icolor = 0
izorder = 0
for i,method in enumerate(['LOG', 'EXP']):
    
    if i == 0:
        a_list = [1.]
        
    else:
        a_list = [5e-2,1e-1,5e-1,1.,1.5]
        
    for ia, a in enumerate(a_list):
        if i == 0:
            label = method
        else:
            label = method + ' %.2f'%a
        data = np.load('result/res_russell_%s_%.2f_n.npz'%(method,a))
        score_test_list, ws_test_list, test_date, _, ws = data['score_test_list'], data['ws_test_list'], data['test_date'], data['n_ws'], data['ws_test_all']
        ws = np.nan_to_num(ws, nan=0.0)
        ws_test_list = np.zeros((len(id_recal), d, len(n_ws)))
        score_test_list = np.zeros((len(test_date), len(n_ws)))
 
        for k in range(len(id_recal)):            
            X_test = df_hold.iloc[id_recal[k]:id_test_end[k]+1,:].values[:,id_codes_list[k]].copy()
            for j, n_w in enumerate(n_ws):
                id_lams = np.where((np.sum(ws[k,:,:] > 0., axis=0) > 0) & (np.sum(ws[k,:,:] > 0., axis=0) <= n_w))[0]
                if len(id_lams) > 0:        
                    w = ws[k, id_codes_list[k], id_lams[-1]]
                    score_test_list[id_recal[k]-id_begin:id_test_end[k]-id_begin+1, j] = np.dot(X_test, w) - 1.
                    ws_test_list[k, id_codes_list[k], j] = w
                else:
                    ws_test_list[k, id_codes_list[k], j] = 0.
        
        cum_ret = np.nancumprod(score_test_list+1,axis=0)
        returns = cum_ret[-1, :] - 1
        risks = np.nanstd(score_test_list, axis=0)

        rf = df_rf.loc[df_rf['date'].isin(test_date), 'rf'].values.reshape((-1,1))

        max_drawdown = np.max((np.maximum.accumulate(cum_ret, axis=0) - cum_ret)/np.maximum.accumulate(cum_ret, axis=0), axis=0)
        
        sharpe_ratio = np.nanmean(np.log(1+score_test_list) - np.log(1+rf), axis=0)/np.nanstd(np.log(1+score_test_list), axis=0)
        
        n_ws_ = np.mean(np.sum(ws_test_list>0., axis=1), axis=0)

        i_sr = np.argmax(sharpe_ratio)
        
    
        axes[0,0].plot(n_ws, returns, ls[ia], marker='^', 
                       c=colors[icolor], label=label, zorder=6-izorder)
        axes[0,1].plot(n_ws, max_drawdown, ls[ia], marker='^', 
                       c=colors[icolor], label=label, zorder=6-izorder)
        axes[1,0].plot(n_ws, sharpe_ratio, ls[ia], marker='^', 
                       c=colors[icolor], label=label, zorder=6-izorder)
        axes[1,1].plot(n_ws, n_ws_, ls[ia], marker='^', 
                       c=colors[icolor], label=label, zorder=6-izorder)
        icolor += 1
        izorder += 1
        
titles = ['Return', 'Max Drawdown', 'Sharpe Ratio', 'Average Number of Assets'] 
for i in range(2):
    for j in range(2):
        axes[i,j].set_title(titles[2*i+j], fontsize=20)
        
        if i==1:
            axes[i,j].set_xlabel('$\|\|w\|\|_0$', fontsize=16)
plt.tight_layout()
fig.subplots_adjust(right=0.9)
axes[1,1].legend(bbox_to_anchor=(1.25, 1.3), loc='upper right', ncol=1, frameon=True, shadow=False, 
                 framealpha=0.3, columnspacing=4, title='Model', fontsize=12)
for i in range(2):
    for j in range(2):
        axes[i,j].patch.set_alpha(0.5)
        axes[i,j].tick_params(axis='x', labelsize=12)
        axes[i,j].tick_params(axis='y', labelsize=12)
plt.savefig('img/russell2000_n.png', dpi=1200, bbox_inches='tight', pad_inches=0)


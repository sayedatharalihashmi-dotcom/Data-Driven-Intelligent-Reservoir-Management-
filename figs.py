import json,numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg');import matplotlib.pyplot as plt
from common import *
plt.rcParams.update({'font.family':'serif','font.size':9})
d=load();info=json.load(open('results.json'));tp=pd.read_csv('test_predictions.csv',parse_dates=['date'])
# data-quality stats
q=d.q.values;print('zeros',(q==0).sum(),'consec-equal nonzero',((q[1:]==q[:-1])&(q[1:]>0)).sum(), 'by year',d.date.dt.year.value_counts().sort_index().to_dict())
# Fig1: real time axis, gaps visible
s=d.set_index('date').q.reindex(pd.date_range('2015-01-01','2020-11-30'))
fig,ax=plt.subplots(figsize=(6.8,2.9))
ax.plot(s.index,s.values,color='#1f4e9c',lw=0.9,marker='.',ms=1.5)
ax.axvspan(pd.Timestamp('2016-01-01'),pd.Timestamp('2018-12-31'),color='#cccccc',alpha=.5,lw=0)
ax.text(pd.Timestamp('2017-07-01'),1500,'No observations\n(2016-2018)',ha='center',va='center',fontsize=8)
ts=pd.Timestamp(info['test_range'][0]);ax.axvspan(ts,pd.Timestamp('2020-11-30'),color='#f4b183',alpha=.5,lw=0)
ax.annotate('Test\nperiod',(ts,1100),xytext=(pd.Timestamp('2019-12-01'),1350),fontsize=8,arrowprops=dict(arrowstyle='->',lw=.6))
ax.set_ylabel('Discharge (m$^3$/s)');ax.set_xlabel('Date');ax.set_xlim(pd.Timestamp('2015-01-01'),pd.Timestamp('2020-12-31'));ax.set_ylim(0,1900)
ax.grid(alpha=.3,lw=.4);fig.tight_layout();fig.savefig('fig1_discharge.png',dpi=220);plt.close()
# Fig2: obs vs LSTM (seed 42) test period
fig,ax=plt.subplots(figsize=(6.8,2.9))
ax.plot(tp.date,tp.observed,color='#222222',lw=1,label='Observed')
ax.plot(tp.date,tp.LSTM,color='#d95f02',lw=1,ls='--',label='LSTM (seed 42)')
ax.set_ylabel('Discharge (m$^3$/s)');ax.set_xlabel('Date');ax.legend(frameon=False);ax.grid(alpha=.3,lw=.4)
fig.autofmt_xdate();fig.tight_layout();fig.savefig('fig2_obs_vs_lstm.png',dpi=220);plt.close()
# Fig3: RMSE bars (LSTM = mean +/- sd over 5 seeds)
R=info['results'];names=['Persistence','ARIMA(2,1,2)','SVR','RF','LSTM'];vals=[R[k]['RMSE'] for k in names[:-1]]+[info['lstm_mean']['RMSE']]
err=[0,0,0,0,info['lstm_std']['RMSE']];cols=['#9e9e9e']*4+['#d95f02']
fig,ax=plt.subplots(figsize=(4.8,2.9));b=ax.bar(names,vals,yerr=err,capsize=4,color=cols,edgecolor='k',lw=.5)
for r,v in zip(b,vals): ax.text(r.get_x()+r.get_width()/2,v+3,f'{v:.1f}',ha='center',va='bottom',fontsize=8)
ax.set_ylabel('Test RMSE (m$^3$/s)');ax.set_ylim(0,140);ax.grid(axis='y',alpha=.3,lw=.4);plt.xticks(rotation=15)
fig.tight_layout();fig.savefig('fig3_rmse.png',dpi=220);plt.close()

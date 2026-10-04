import json,time,numpy as np,pandas as pd
from scipy.optimize import least_squares
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from common import *
import lstm_np
d=load();q=d.q.values;dates=d.date.values;n=len(q);L=14
seg=np.cumsum(d.date.diff().dt.days.fillna(1).values>1)
# supervised samples only from windows with 15 consecutive calendar days
S=[]
for i in range(L,n):
    if seg[i-L]==seg[i]: S.append(i)
S=np.array(S);X=np.array([q[i-L:i] for i in S]);y=q[S];sd=dates[S];ssg=seg[S]
N=len(S);a=int(N*.7);b=int(N*.85)
print('windows',N,'train',a,'val',b-a,'test',N-b)
info=dict(N=N,ntr=a,nva=b-a,nte=N-b,train_range=[str(sd[0])[:10],str(sd[a-1])[:10]],val_range=[str(sd[a])[:10],str(sd[b-1])[:10]],test_range=[str(sd[b])[:10],str(sd[-1])[:10]],
          test_mean=float(y[b:].mean()),test_max=float(y[b:].max()),test_std=float(y[b:].std()),train_max=float(y[:a].max()),val_max=float(y[a:b].max()),
          segments_total=int(seg.max()+1),segments_ge15=int(sum((seg==s).sum()>=15 for s in np.unique(seg))),n_obs=n)
print(info)
Xtr,ytr,Xva,yva,Xte,yte=X[:a],y[:a],X[a:b],y[a:b],X[b:],y[b:]
mu,sg=Xtr.mean(),Xtr.std()   # scalar standardisation on training data
z=lambda v:(v-mu)/sg;iz=lambda v:v*sg+mu
R={};P={}
# persistence
P['Persistence']=Xte[:,-1];R['Persistence']=metrics(yte,P['Persistence'])
# SVR (paper config) + validation grid for justification
def svr(C,e): return SVR(kernel='rbf',C=C,epsilon=e).fit(z(Xtr),z(ytr))
m=svr(10,0.01);P['SVR']=iz(m.predict(z(Xte)));R['SVR']=metrics(yte,P['SVR'])
grid={}
for C in [1,10,100]:
    for e in [0.01,0.1]:
        mm=svr(C,e);grid[f'C={C},eps={e}']=float(np.sqrt(np.mean((iz(mm.predict(z(Xva)))-yva)**2)))
info['svr_val_grid_rmse']=grid
# RF
rf=RandomForestRegressor(300,random_state=42,n_jobs=1,min_samples_leaf=2).fit(Xtr,ytr);P['RF']=rf.predict(Xte);R['RF']=metrics(yte,P['RF'])
# ARIMA(p,1,q) CSS, per contiguous segment, no constant
def resid(par,p,qq,series):
    ph=par[:p];th=par[p:p+qq];out=[]
    for w in series:
        e=np.zeros(len(w))
        for t in range(len(w)):
            ar=sum(ph[k]*w[t-1-k] for k in range(p) if t-1-k>=0);ma=sum(th[k]*e[t-1-k] for k in range(qq) if t-1-k>=0)
            e[t]=w[t]-ar-ma
        out.append(e)
    return np.concatenate(out)
test_start=sd[b]
segs=[];
for s in np.unique(seg):
    idx=np.where(seg==s)[0];idx=idx[dates[idx]<test_start]
    if len(idx)>=3: segs.append(np.diff(q[idx]))
ar={}
for p in range(3):
    for qq in range(3):
        if p==0 and qq==0: continue
        r=least_squares(resid,np.zeros(p+qq)+0.05,args=(p,qq,segs),bounds=(-0.98,0.98));sse=(r.fun**2).sum();k=len(r.fun)
        ar[(p,qq)]=(r.x,k*np.log(sse/k)+2*(p+qq))
info['arima_aic']={f'({p},1,{qq})':float(v[1]) for (p,qq),v in ar.items()}
par=ar[(2,2)][0];info['arima_params']=[float(x) for x in par]
# rolling one-step forecast with fixed params over each segment containing test targets
pred={}
for s in np.unique(seg):
    idx=np.where(seg==s)[0]
    if len(idx)<2: continue
    w=np.diff(q[idx]);e=np.zeros(len(w));ph,th=par[:2],par[2:]
    for t in range(len(w)):
        arv=sum(ph[k]*w[t-1-k] for k in range(2) if t-1-k>=0);mav=sum(th[k]*e[t-1-k] for k in range(2) if t-1-k>=0)
        pred[idx[t+1]]=q[idx[t]]+arv+mav;e[t]=w[t]-arv-mav
P['ARIMA(2,1,2)']=np.array([pred[i] for i in S[b:]]);R['ARIMA(2,1,2)']=metrics(yte,P['ARIMA(2,1,2)'])
# LSTM, 5 seeds
Xs=lambda A:z(A)[:,:,None]
runs={}
for sd_ in [42,43,44,45,46]:
    t0=time.time();net=lstm_np.fit(Xs(Xtr),z(ytr),Xs(Xva),z(yva),seed=sd_)
    p=iz(net.predict(Xs(Xte)));runs[sd_]=dict(m=metrics(yte,p),ep=net.epochs,p=p);print('lstm',sd_,net.epochs,round(time.time()-t0),'s',{k:round(float(v),3) for k,v in runs[sd_]['m'].items()},flush=True)
P['LSTM']=runs[42]['p'];R['LSTM']=runs[42]['m']
ks=list(R['LSTM'].keys())
info['lstm_seeds']={str(k):{kk:float(v) for kk,v in r['m'].items()} for k,r in runs.items()}
info['lstm_mean']={k:float(np.mean([r['m'][k] for r in runs.values()])) for k in ks}
info['lstm_std']={k:float(np.std([r['m'][k] for r in runs.values()],ddof=1)) for k in ks}
info['lstm_epochs']={str(k):r['ep'] for k,r in runs.items()}
info['results']={m:{k:float(v) for k,v in r.items()} for m,r in R.items()}
for m,r in info['results'].items(): print(m,{k:round(v,3) for k,v in r.items()})
print('LSTM mean',{k:round(v,3) for k,v in info['lstm_mean'].items()});print('LSTM std',{k:round(v,3) for k,v in info['lstm_std'].items()})
print('arima aic',info['arima_aic']);print('svr grid',grid)
json.dump(info,open('results.json','w'),indent=1)
pd.DataFrame({'date':pd.to_datetime(sd[b:]),'observed':yte,**{k:v for k,v in P.items()},**{f'LSTM_seed{k}':r['p'] for k,r in runs.items()}}).to_csv('test_predictions.csv',index=False)

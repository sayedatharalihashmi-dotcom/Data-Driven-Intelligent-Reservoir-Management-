import numpy as np
def sig(x): return 1/(1+np.exp(-x))
class LSTMLayer:
    def __init__(s,D,H,rng):
        lim=np.sqrt(6/(D+4*H)); s.Wx=rng.uniform(-lim,lim,(D,4*H))
        Wh=[]
        for _ in range(4):
            a=rng.standard_normal((H,H)); q,r=np.linalg.qr(a); Wh.append(q*np.sign(np.diag(r)))
        s.Wh=np.concatenate(Wh,1); s.b=np.zeros(4*H); s.b[H:2*H]=1.0; s.H=H
    def forward(s,X):
        B,T,D=X.shape;H=s.H;h=np.zeros((B,H));c=np.zeros((B,H));Hs=np.zeros((B,T,H));cache=[]
        for t in range(T):
            a=X[:,t]@s.Wx+h@s.Wh+s.b
            i=sig(a[:,:H]);f=sig(a[:,H:2*H]);g=np.tanh(a[:,2*H:3*H]);o=sig(a[:,3*H:])
            cn=f*c+i*g;tc=np.tanh(cn);hn=o*tc
            cache.append((X[:,t],h,c,i,f,g,o,tc));h,c=hn,cn;Hs[:,t]=h
        s.cache=cache;return Hs
    def backward(s,dHs):
        B,T,H=dHs.shape;dWx=np.zeros_like(s.Wx);dWh=np.zeros_like(s.Wh);db=np.zeros_like(s.b)
        dX=np.zeros((B,T,s.Wx.shape[0]));dh=np.zeros((B,H));dc=np.zeros((B,H))
        for t in reversed(range(T)):
            x,hp,cp,i,f,g,o,tc=s.cache[t]
            dh=dh+dHs[:,t];do=dh*tc;dct=dh*o*(1-tc**2)+dc
            di=dct*g;dg=dct*i;df=dct*cp;dc=dct*f
            dA=np.concatenate([di*i*(1-i),df*f*(1-f),dg*(1-g**2),do*o*(1-o)],1)
            dWx+=x.T@dA;dWh+=hp.T@dA;db+=dA.sum(0);dX[:,t]=dA@s.Wx.T;dh=dA@s.Wh.T
        return dX,[dWx,dWh,db]
class Net:
    def __init__(s,D=1,H=32,drop=0.1,seed=42):
        rng=np.random.default_rng(seed);s.rng=rng;s.l1=LSTMLayer(D,H,rng);s.l2=LSTMLayer(H,H,rng)
        lim=np.sqrt(6/(H+1));s.Wo=rng.uniform(-lim,lim,(H,1));s.bo=np.zeros(1);s.drop=drop
    def params(s): return [s.l1.Wx,s.l1.Wh,s.l1.b,s.l2.Wx,s.l2.Wh,s.l2.b,s.Wo,s.bo]
    def forward(s,X,train=False):
        H1=s.l1.forward(X)
        if train and s.drop>0:
            s.mask=(s.rng.random(H1.shape)>=s.drop)/(1-s.drop);H1=H1*s.mask
        else: s.mask=None
        H2=s.l2.forward(H1);s.last=H2[:,-1];return (s.last@s.Wo+s.bo)[:,0],H2
    def loss_grads(s,X,y):
        p,H2=s.forward(X,True);B=len(y);err=p-y;loss=np.mean(err**2)
        dp=2*err/B;dWo=s.last.T@dp[:,None];dbo=np.array([dp.sum()])
        dH2=np.zeros_like(H2);dH2[:,-1]=dp[:,None]@s.Wo.T
        dH1,g2=s.l2.backward(dH2)
        if s.mask is not None: dH1=dH1*s.mask
        _,g1=s.l1.backward(dH1)
        return loss,g1+g2+[dWo,dbo]
    def predict(s,X): return s.forward(X,False)[0]
def fit(Xtr,ytr,Xva,yva,seed=42,H=32,drop=0.1,lr=1e-3,bs=16,patience=12,max_ep=120):
    net=Net(1,H,drop,seed);P=net.params();m=[np.zeros_like(p) for p in P];v=[np.zeros_like(p) for p in P]
    b1,b2,eps=0.9,0.999,1e-7;step=0;best=np.inf;bestP=None;wait=0;rng=np.random.default_rng(seed);ep_run=0
    for ep in range(max_ep):
        idx=rng.permutation(len(ytr))
        for k in range(0,len(idx),bs):
            j=idx[k:k+bs];loss,G=net.loss_grads(Xtr[j],ytr[j]);step+=1
            for n,(p,g) in enumerate(zip(P,G)):
                m[n]=b1*m[n]+(1-b1)*g;v[n]=b2*v[n]+(1-b2)*g*g
                p-=lr*(m[n]/(1-b1**step))/(np.sqrt(v[n]/(1-b2**step))+eps)
        vl=np.mean((net.predict(Xva)-yva)**2);ep_run=ep+1
        if vl<best-1e-8: best=vl;bestP=[p.copy() for p in P];wait=0
        else:
            wait+=1
            if wait>=patience: break
    for p,b in zip(P,bestP): p[...]=b
    net.epochs=ep_run;return net
if __name__=='__main__':
    rng=np.random.default_rng(0);X=rng.standard_normal((4,5,1));y=rng.standard_normal(4)
    net=Net(1,6,0.0,1);loss,G=net.loss_grads(X,y);P=net.params();mx=0
    for pi,(p,g) in enumerate(zip(P,G)):
        for _ in range(6):
            ix=tuple(rng.integers(0,s) for s in p.shape);o=p[ix];e=1e-6
            p[ix]=o+e;lp=np.mean((net.forward(X)[0]-y)**2);p[ix]=o-e;lm=np.mean((net.forward(X)[0]-y)**2);p[ix]=o
            num=(lp-lm)/(2*e);mx=max(mx,abs(num-g[ix])/(abs(num)+abs(g[ix])+1e-12))
    print('max rel grad err',mx)

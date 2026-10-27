import numpy as np
K=3; C_CH=8; L=64; DILS_MANY=[1,2,4,8,16,32]

def causal_dconv(x,W,b,d):
    B,cin,Lx=x.shape; cout=W.shape[0]; out=np.zeros((B,cout,Lx))
    for k in range(K):
        idx=np.arange(Lx)-k*d; valid=idx>=0; taps=np.zeros_like(x); taps[:,:,valid]=x[:,:,idx[valid]]
        for c in range(cout):
            out[:,c,:]+=b[c]; out[:,c,:]+=np.tensordot(taps,W[c,:,k],axes=([1],[0]))
    return out
def relu(z): return np.maximum(0,z)
def drelu(z): return (z>0).astype(float)
def init_cnn(seed=0):
    rg=np.random.default_rng(seed); P={}; cin=1
    for i,d in enumerate(DILS_MANY[:5]):
        cout=C_CH; P[f"W{i}"]=rg.standard_normal((cout,cin,K))*0.05; P[f"b{i}"]=np.zeros(cout)
        P[f"Wr{i}"]=np.eye(cout) if cin==cout else rg.standard_normal((cin,cout))*0.1; cin=cout
    P["Wo"]=rg.standard_normal((1,C_CH))*0.1; P["bo"]=np.zeros(1); return P
def forward(X,P,dils):
    h=X; acts=[]
    for i,d in enumerate(dils):
        conv=causal_dconv(h,P[f"W{i}"],P[f"b{i}"],d); res=np.einsum("bcl,cj->bjl",h,P[f"Wr{i}"]); a=relu(conv+res); acts.append((h,conv,res,a)); h=a
    last=h[:,:,-1]; return (last@P["Wo"].T+P["bo"].ravel()).ravel(),acts
def backward(X,y,P,dils,lr=0.02):
    B=X.shape[0]; yh,acts=forward(X,P,dils); dL=(yh-y)/B
    dh=np.zeros((B,C_CH,L)); dh[:,:,-1]=(dL.reshape(-1,1)@P["Wo"]).reshape(B,C_CH)
    gWo=dL.reshape(-1,1).T@acts[-1][3][:,:,-1]; gbo=dL.sum().reshape(1); grads={"Wo":gWo,"bo":gbo}
    for i in reversed(range(len(dils))):
        h,conv,res,a=acts[i]; da=dh*drelu(conv+res)
        gW=np.zeros_like(P[f"W{i}"]); gb=da.sum(axis=(0,2)); gWr=np.einsum("bil,bol->io",h,da)
        for k in range(K):
            idx=np.arange(L)-k*dils[i]; valid=idx>=0; hd=np.zeros_like(h); hd[:,:,valid]=h[:,:,idx[valid]]
            for c in range(C_CH):
                gW[c,:,k]+=np.einsum("bl,bil->i", da[:,c,:], hd)
        dhp=np.einsum("bjl,cj->bcl",da,P[f"Wr{i}"]); dht=np.zeros((B,C_CH,L))
        for k in range(K):
            idx=np.arange(L)-k*dils[i]; valid=idx>=0; gs=np.zeros_like(da); gs[:,:,valid]=da[:,:,idx[valid]]
            dht+=np.einsum("bcl,cj->bjl",gs,P[f"W{i}"][:,:,k])
        dh=dhp+dht; grads[f"W{i}"]=gW; grads[f"b{i}"]=gb; grads[f"Wr{i}"]=gWr
    for k in grads: P[k]-=lr*grads[k]
    return float(np.mean((yh-y)**2))

def make_windows(x,y,T=L):
    X,Y=[],[]
    for i in range(T,len(x)): X.append(x[i-T:i]); Y.append(y[i])
    return np.array(X).reshape(-1,1,T),np.array(Y)

def run(siggen, name, seed=20260723, ep=120):
    rg=np.random.default_rng(seed); x,y=siggen(rg)
    y=(y-y.mean())/(y.std()+1e-9); x=y.copy()
    X,Y=make_windows(x,y); n=6400
    Xtr,Ytr=X[:n],Y[:n]; Xte,Yte=X[n:],Y[n:]
    # OLS
    Xl=np.array([x[i-L:i][::-1] for i in range(L,len(x))]); Yl=y[L:]
    Wl,*_=np.linalg.lstsq(Xl[:n],Yl[:n],rcond=None); pols=1-np.sum((Yl[n:]-Xl[n:]@Wl)**2)/np.sum((Yl[n:]-Yl[n:].mean())**2)
    # plain CNN
    Pp=init_cnn(); DILS=[1,1,1,1,1]
    for _ in range(ep):
        idx=rg.permutation(n)
        for s in range(0,n,64): b=idx[s:s+64]; backward(Xtr[b],Ytr[b],Pp,DILS)
    mp=1-np.sum((Yte-forward(Xte,Pp,DILS)[0])**2)/np.sum((Yte-Yte.mean())**2)
    # dilated
    Pt=init_cnn(); DILS=[1,2,4,8,16]
    for _ in range(ep):
        idx=rg.permutation(n)
        for s in range(0,n,64): b=idx[s:s+64]; backward(Xtr[b],Ytr[b],Pt,DILS)
    md=1-np.sum((Yte-forward(Xte,Pt,DILS)[0])**2)/np.sum((Yte-Yte.mean())**2)
    print(f"{name:38s} OLS_R2={pols:.3f}  plainCNN_R2={mp:.3f}  dilCNN_R2={md:.3f}  (dil-plain)={md-mp:+.3f}")

def s_sin(rg):
    t=np.arange(8000); y=np.zeros(8000)
    for f in (5,11,17,23,31):
        y+=rg.normal(0,1)*0.8*np.sin(2*np.pi*f*t/64)+rg.normal(0,1)*0.8*np.cos(2*np.pi*f*t/64)
    return None,y+0.05*rg.standard_normal(8000)

def s_sinmod(rg):
    t=np.arange(8000); y=np.sin(2*np.pi*13*t/64)
    env=0.4+0.6*(0.5+0.5*np.sin(2*np.pi*t/256)); y*=env
    y+=0.05*rg.standard_normal(8000); return None,y

def s_sq(rg):
    t=np.arange(8000); base=np.sin(2*np.pi*11*t/64)
    y=np.where(base>=0,1.0,-1.0); y+=0.15*rg.standard_normal(8000); return None,y

def s_longsin(rg):
    t=np.arange(8000); y=np.sin(2*np.pi*t/40)+0.5*np.sin(2*np.pi*t/128); return None,y+0.05*rg.standard_normal(8000)

def s_mix(rg):
    t=np.arange(8000); y=np.sin(2*np.pi*9*t/64)+np.sign(np.sin(2*np.pi*9*t/64))*0.3*np.cos(2*np.pi*23*t/64)
    # fold: |sin| removed -> still periodic 40; add a frequency that only appears every 3 cycles
    y+=0.4*np.sin(2*np.pi*t/64)* (0.5+0.5*np.cos(2*np.pi*t/192)); return None,y+0.05*rg.standard_normal(8000)

def s_nl_period(rg):
    # y_t = sin(2π t/40) * g(sin(2π t/128)) — global slow nonlinearity; local view can't recover
    t=np.arange(8000); y=np.sin(2*np.pi*t/40)*(0.5+0.5*np.sin(2*np.pi*t/128))
    return None,y+0.03*rg.standard_normal(8000)

run(s_sin,"pure multi-sine")
run(s_sinmod,"sine with 256-step envelope")
run(s_sq,"square wave (11)")
run(s_longsin,"sine 40 + sine 128")
run(s_mix,"mix fold")
run(s_nl_period,"nonlinear period (40 x env 128)")

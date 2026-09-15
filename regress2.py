"""Regression benchmark with repeated k-fold cross-validation.

At n = 26 a single leave-one-out estimate is unstable, so every model is
evaluated by 5-fold cross-validation repeated 50 times with independent random
partitions. Standardisation and ridge-penalty selection (by generalised
cross-validation) are carried out inside each training fold. We report the mean
and the 5th-95th percentile range of Q^2 over repeats, and a paired comparison
of squared errors between descriptor sets on identical partitions.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import sys, json, os
import numpy as np, pandas as pd
from scipy import stats

R=str(paths.RESULTS)
A=json.load(open(f'{R}/setA.json')); B=json.load(open(f'{R}/setB.json'))
if os.path.exists(f'{R}/esp_axial.json'):
    for n,v in json.load(open(f'{R}/esp_axial.json')).items():
        if n in A: A[n].update(v)

BASE = ['nheavy','n_aromatic_rings','vdw_area','tpsa','esp_min','esp_max','esp_var','mw',
        'esp_axial_2.0','esp_axial_2.6','esp_axial_3.2']
NPI  = ['Npi_grid']
IDT  = ['piIDT0_n','piIDT0_totpers','piIDT0_maxpers','piIDT0_entropy',
        'piIDT1_n','piIDT1_totpers','piIDT1_maxpers','piIDT1_entropy',
        'piIDT2_n','piIDT2_totpers','Api_integral','occlusion_ratio','A_pi_max']
COMPACT = ['Api_integral','occlusion_ratio','piIDT1_totpers','piIDT0_totpers','A_pi_max']
TARGETS=['eint_Na+','eint_Cl-','eint_benzene','eint_CH4']
ALPHAS=np.logspace(-2,5,60)
NREP, KFOLD = 50, 5

rows=[]
for n,r in B.items():
    if n not in A: continue
    d={'name':n}
    d.update({k:A[n][k] for k in set(BASE+NPI+IDT+COMPACT) if k in A[n]})
    for t in TARGETS: d[t]=r.get(t,np.nan)
    rows.append(d)
D=pd.DataFrame(rows).set_index('name'); D.to_csv(f'{R}/setB_table.csv')

def fit_predict(Xtr,ytr,Xte):
    mu=Xtr.mean(0); sd=Xtr.std(0); sd[sd<1e-12]=1.0
    Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
    ym=ytr.mean(); yc=ytr-ym
    U,s,Vt=np.linalg.svd(Xtr,full_matrices=False); s2=s**2; Uty=U.T@yc
    n=len(ytr); best=None
    for a in ALPHAS:
        d=s2/(s2+a); yhat=U@(d*Uty); tr=d.sum()+1.0
        gcv=n*float(((yc-yhat)**2).sum())/max((n-tr)**2,1e-12)
        if best is None or gcv<best[0]: best=(gcv,a)
    a=best[1]
    w=Vt.T@((s/(s2+a))*Uty)
    return ym+Xte@w, a

def repeated_cv(X,y,seed=0):
    n=len(y); preds=np.zeros((NREP,n)); alphas=[]
    for rep in range(NREP):
        rng=np.random.default_rng(1000*seed+rep); idx=rng.permutation(n)
        folds=np.array_split(idx,KFOLD)
        for f in folds:
            tr=np.setdiff1d(idx,f)
            p,a=fit_predict(X[tr],y[tr],X[f]); preds[rep,f]=p; alphas.append(a)
    q2=np.array([1-((y-preds[r])**2).sum()/((y-y.mean())**2).sum() for r in range(NREP)])
    rmse=np.array([np.sqrt(((y-preds[r])**2).mean()) for r in range(NREP)])
    return dict(q2_mean=float(q2.mean()), q2_lo=float(np.percentile(q2,5)),
                q2_hi=float(np.percentile(q2,95)), q2_sd=float(q2.std()),
                rmse_mean=float(rmse.mean()),
                r=float(stats.pearsonr(y,preds.mean(0))[0]),
                alpha_med=float(np.median(alphas)),
                pred=preds.mean(0).tolist(), sqerr=((y-preds)**2).mean(0).tolist())

def perm_p(X,y,q2,nperm=199,seed=7):
    rng=np.random.default_rng(seed); c=0
    for i in range(nperm):
        if repeated_cv(X,rng.permutation(y),seed=100+i)['q2_mean']>=q2: c+=1
    return (c+1)/(nperm+1)

SETS={'baseline':BASE,'baseline+Npi':BASE+NPI,'piIDT only':IDT,
      'piIDT compact':COMPACT,'baseline+compact':BASE+COMPACT,'baseline+piIDT':BASE+IDT}
out={}
for t in TARGETS:
    sub=D.dropna(subset=[t]); y=sub[t].values
    out[t]={'n':int(len(sub)),'y_sd':float(y.std(ddof=1)),'names':sub.index.tolist(),
            'y':y.tolist()}
    for k,cols in SETS.items():
        cols=[c for c in cols if c in sub.columns and sub[c].std()>0]
        X=np.asarray(sub[cols].values,float)
        m=repeated_cv(X,y); m['features']=cols
        m['p_perm']=perm_p(X,y,m['q2_mean']) if k in ('baseline','baseline+piIDT','piIDT only') else None
        out[t][k]=m
        print('%-14s %-17s p=%2d  Q2=%6.3f [%6.3f,%6.3f]  RMSE=%6.3f  r=%5.3f  perm=%s'%(
            t,k,len(cols),m['q2_mean'],m['q2_lo'],m['q2_hi'],m['rmse_mean'],m['r'],
            '%.3f'%m['p_perm'] if m['p_perm'] is not None else '  -- '), flush=True)
    rng=np.random.default_rng(3)
    e0=np.array(out[t]['baseline']['sqerr']); e1=np.array(out[t]['baseline+piIDT']['sqerr'])
    dif=np.array([ (e0[i]-e1[i]).mean() for i in
                   (rng.integers(0,len(y),len(y)) for _ in range(5000)) ])
    out[t]['boot_gain']=dict(mean=float(dif.mean()),lo=float(np.percentile(dif,2.5)),
                             hi=float(np.percentile(dif,97.5)),p_gt0=float((dif>0).mean()))
    print('   paired bootstrap MSE gain from pi-IDT: %.3f [%.3f, %.3f]  P(gain>0)=%.3f'%(
        dif.mean(),np.percentile(dif,2.5),np.percentile(dif,97.5),(dif>0).mean()), flush=True)
json.dump(out, open(f'{R}/regression.json','w'), indent=1)

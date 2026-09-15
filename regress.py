"""Regression benchmark: do pi-IDT descriptors add signal over baselines?

Ridge regression with an analytic leave-one-out estimator (SVD + hat matrix),
the penalty chosen by generalised cross-validation on the training data. The
same estimator is used for the permutation null, so the p-values are internally
consistent.
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
ALPHAS=np.logspace(-3,4,60)

rows=[]
for n,r in B.items():
    if n not in A: continue
    d={'name':n}
    d.update({k:A[n][k] for k in BASE+NPI+IDT+COMPACT if k in A[n]})
    for t in TARGETS: d[t]=r.get(t,np.nan)
    rows.append(d)
D=pd.DataFrame(rows).set_index('name'); D.to_csv(f'{R}/setB_table.csv')

def prep(X):
    X=np.asarray(X,float); mu=X.mean(0); sd=X.std(0); sd[sd<1e-12]=1.0
    return (X-mu)/sd

def loo_ridge(Xs, y):
    """Analytic LOO predictions; alpha selected by GCV on the full data."""
    n=len(y); yc=y-y.mean()
    U,s,Vt=np.linalg.svd(Xs, full_matrices=False)
    Uty=U.T@yc; s2=s**2
    best=None
    for a in ALPHAS:
        d=s2/(s2+a)
        yhat=U@(d*Uty)
        tr=d.sum()+1.0
        rss=float(((yc-yhat)**2).sum())
        gcv=n*rss/max((n-tr)**2,1e-12)
        if best is None or gcv<best[0]: best=(gcv,a,d,yhat)
    _,a,d,yhat=best
    h=(U**2)@d + 1.0/n
    pred=y.mean()+yhat-(yc-yhat)*0  # placeholder, replaced below
    loo=yc-(yc-yhat)/np.clip(1-h,1e-6,None)
    pred=y.mean()+loo
    ss_res=float(((y-pred)**2).sum()); ss_tot=float(((y-y.mean())**2).sum())
    return dict(q2=1-ss_res/ss_tot, rmse=float(np.sqrt(((y-pred)**2).mean())),
                mae=float(np.abs(y-pred).mean()),
                r=float(stats.pearsonr(y,pred)[0]), alpha=float(a),
                pred=pred.tolist())

def perm_p(Xs,y,q2,nperm=999,seed=0):
    rng=np.random.default_rng(seed); c=0
    for _ in range(nperm):
        if loo_ridge(Xs,rng.permutation(y))['q2']>=q2: c+=1
    return (c+1)/(nperm+1)

out={}
for t in TARGETS:
    sub=D.dropna(subset=[t]); y=sub[t].values
    sets={'baseline':BASE,'baseline+Npi':BASE+NPI,'piIDT only':IDT,
          'baseline+piIDT':BASE+IDT,'piIDT compact':COMPACT,
          'baseline+compact':BASE+COMPACT}
    out[t]={'n':int(len(sub)),'y_mean':float(y.mean()),'y_sd':float(y.std(ddof=1)),
            'names':sub.index.tolist(),'y':y.tolist()}
    Xs_cache={}
    for k,cols in sets.items():
        cols=[c for c in cols if c in sub.columns and sub[c].std()>0]
        Xs=prep(sub[cols].values); Xs_cache[k]=Xs
        m=loo_ridge(Xs,y); m['p_perm']=perm_p(Xs,y,m['q2']); m['features']=cols
        out[t][k]=m
        print('%-14s %-16s n=%2d p=%2d  Q2=%6.3f  RMSE=%6.3f  r=%5.3f  perm p=%.3f'%(
            t,k,len(sub),len(cols),m['q2'],m['rmse'],m['r'],m['p_perm']), flush=True)
    # paired bootstrap on the LOO residuals: does adding pi-IDT reduce error?
    rng=np.random.default_rng(1)
    e0=(y-np.array(out[t]['baseline']['pred']))**2
    e1=(y-np.array(out[t]['baseline+piIDT']['pred']))**2
    e2=(y-np.array(out[t]['baseline+compact']['pred']))**2
    dif=[]
    for _ in range(5000):
        i=rng.integers(0,len(y),len(y)); dif.append(e0[i].mean()-e1[i].mean())
    dif=np.array(dif)
    out[t]['boot_mse_gain']=dict(mean=float(dif.mean()),
        lo=float(np.percentile(dif,2.5)), hi=float(np.percentile(dif,97.5)),
        p_gt0=float((dif>0).mean()))
    rng=np.random.default_rng(2); dif2=[]
    for _ in range(5000):
        i=rng.integers(0,len(y),len(y)); dif2.append(e0[i].mean()-e2[i].mean())
    dif2=np.array(dif2)
    out[t]['boot_mse_gain_compact']=dict(mean=float(dif2.mean()),
        lo=float(np.percentile(dif2,2.5)), hi=float(np.percentile(dif2,97.5)),
        p_gt0=float((dif2>0).mean()))
    print('   bootstrap MSE gain  full: %.3f [%.3f, %.3f] P=%.3f | compact: %.3f [%.3f, %.3f] P=%.3f'%(
        dif.mean(),np.percentile(dif,2.5),np.percentile(dif,97.5),(dif>0).mean(),
        dif2.mean(),np.percentile(dif2,2.5),np.percentile(dif2,97.5),(dif2>0).mean()), flush=True)
json.dump(out, open(f'{R}/regression_loo.json','w'), indent=1)

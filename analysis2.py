"""Robustness, regression figures/tables and the numbers macro file."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import sys, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':8,'axes.linewidth':0.6,'legend.frameon':False,
                     'font.family':'serif','mathtext.fontset':'dejavuserif',
                     'savefig.bbox':'tight','savefig.pad_inches':0.02})
R=str(paths.RESULTS); P=str(paths.ANALYSIS)
A=json.load(open(f'{R}/setA.json'))
NUM=json.load(open(f'{R}/numbers_partial.json')) if os.path.exists(f'{R}/numbers_partial.json') else {}
def N(k,v): NUM[k]=v; return v
def esc(s): return s.replace('_','\\_').replace('[','{[}').replace(']','{]}')

# ---------------------------------------------------------------- partition
sym=[r for r in A.values() if r['partition']=='symmetry']
err=[abs(r['Npi_sym']-round(r['Npi_sym'])) for r in sym]
leak=[r['sigma_pi_leakage'] for r in sym]
def sci(x, d=1):
    m, e = ('%.*e'%(d, x)).split('e')
    return r'$%s\times10^{%d}$'%(m, int(e))
N('nSym',len(sym))
N('maxNpiErr',sci(max(err))); N('maxLeak',sci(max(leak)))
PV=json.load(open(f'{R}/poav_v2.json')) if os.path.exists(f'{R}/poav_v2.json') else {}
cmp_=[(n,v['reldiff'],v['Npi_poav'],v['Npi_sym']) for n,v in PV.items()]
if cmp_:
    N('nPoavCmp',len(cmp_))
    N('poavRelMean','%.1f'%(100*np.mean([c[1] for c in cmp_])))
    N('poavRelMax','%.1f'%(100*np.max([c[1] for c in cmp_])))
    N('poavRelMin','%.1f'%(100*np.min([c[1] for c in cmp_])))
    N('poavNpiMAE','%.3f'%np.mean([abs(c[2]-c[3]) for c in cmp_]))
    N('poavNpiMax','%.3f'%np.max([abs(c[2]-c[3]) for c in cmp_]))
    ex=[c for c in cmp_ if c[0]!='benzonitrile']
    N('poavRelMeanEx','%.1f'%(100*np.mean([c[1] for c in ex])))
    N('poavRelMaxEx','%.1f'%(100*np.max([c[1] for c in ex])))

# --------------------------------------------------------------- robustness
if os.path.exists(f'{R}/robust.json'):
    RB=json.load(open(f'{R}/robust.json'))
    KEYS=['piIDT0_n','piIDT1_n','piIDT1_totpers','piIDT0_totpers','Api_integral']
    LBL={'piIDT0_n':r'$\beta_0$','piIDT1_n':r'$\beta_1$',
         'piIDT1_totpers':r'$T_1$','piIDT0_totpers':r'$T_0$',
         'Api_integral':r'$I_\pi$'}
    rows=[]
    def spread(vals):
        v=np.array(vals,float)
        return (v.max()-v.min())/abs(np.mean(v)) if abs(np.mean(v))>1e-12 else 0.0
    for nm,rec in RB.items():
        r0=rec['ref_poav']
        rot=[d for d in rec['rot']]
        for k in KEYS:
            vals=[r0[k]]+[d[k] for d in rot]
            rows.append(dict(molecule=nm,test='rotation',key=k,
                             rel_range=spread(vals), ref=r0[k]))
        for test,dct in [('grid',rec.get('grid')),('padding',rec.get('pad')),
                         ('basis',rec.get('basis')),('functional',rec.get('xc')),
                         ('mask',rec.get('mask'))]:
            if not dct: continue
            sub=dct
            if test=='grid': sub={kk:v for kk,v in dct.items() if float(kk)<=0.20}
            for k in KEYS:
                vals=[v[k] for v in sub.values()]
                rows.append(dict(molecule=nm,test=test,key=k,
                                 rel_range=spread(vals), ref=rec['ref'][k]))
        for j,lst in rec.get('jitter',{}).items():
            for k in KEYS:
                vals=[rec['ref_poav'][k]]+[d[k] for d in lst]
                rows.append(dict(molecule=nm,test=f'jitter {j} AA',key=k,
                                 rel_range=spread(vals), ref=rec['ref_poav'][k]))
    RD=pd.DataFrame(rows); RD.to_csv(f'{R}/robust_table.csv',index=False)
    piv=RD.pivot_table(index='test',columns='key',values='rel_range',aggfunc='max')
    order=['rotation','grid','padding','basis','functional','mask',
           'jitter 0.01 AA','jitter 0.03 AA','jitter 0.05 AA']
    piv=piv.reindex([o for o in order if o in piv.index])
    lines=[]
    for t,row in piv.iterrows():
        lab=t.replace('_','\\_').replace(' AA', '~\\AA')
        lines.append('%s & %s'%(lab,
              ' & '.join('%.3f'%row[k] if np.isfinite(row[k]) else '--' for k in KEYS)))
    tex=(r'\begin{tabular}{l'+'c'*len(KEYS)+r'}'+'\n\\toprule\nperturbation & '
         +' & '.join(LBL[k] for k in KEYS)+r'\\'+'\n\\midrule\n'
         +'\\\\\n'.join(lines)+'\\\\\n\\bottomrule\n\\end{tabular}')
    open(f'{P}/tables/table_robust.tex','w').write(tex)
    rot_max=RD[RD.test=='rotation'].rel_range.max()
    N('rotMaxRel','%.4f'%rot_max)
    N('betaRotExact', int((RD[(RD.test=='rotation')&(RD.key.isin(['piIDT0_n','piIDT1_n']))].rel_range<1e-9).all()))
    g=RD[RD.test=='grid']
    N('gridMaxRel','%.3f'%g.rel_range.max())
    N('gridMaxRelTone','%.3f'%g[g.key=='piIDT1_totpers'].rel_range.max())
    N('gridMaxRelTzero','%.2f'%g[g.key=='piIDT0_totpers'].rel_range.max())
    ro=RD[RD.test=='rotation']
    N('rotMaxRelTone','%.2f'%ro[ro.key=='piIDT1_totpers'].rel_range.max())
    inv=[k for k in ro.key.unique() if ro[ro.key==k].rel_range.max()<1e-9]
    N('rotInvariantKeys',', '.join(sorted(inv)))
    N('xcMaxRel','%.3f'%RD[RD.test=='functional'].rel_range.max())
    N('basisMaxRel','%.3f'%RD[RD.test=='basis'].rel_range.max())
    mk=RD[RD.test=='mask']
    N('maskMaxRel','%.3f'%mk.rel_range.max())
    N('maskMaxRelI','%.2f'%mk[mk.key=='Api_integral'].rel_range.max())

    # robustness figure
    fig,axs=plt.subplots(1,3,figsize=(7.0,2.1))
    ax=axs[0]
    for nm,rec in RB.items():
        hs=sorted(rec['grid'],key=float)
        ax.plot([float(h) for h in hs],[rec['grid'][h]['piIDT1_totpers'] for h in hs],
                'o-',ms=3,lw=1,label=nm)
    ax.set_xlabel('grid spacing $h$ / \u00c5'); ax.set_ylabel(r'$T_1$ / a.u.')
    ax.set_title('(a) grid convergence',fontsize=7.5); ax.legend(fontsize=5.5)
    ax=axs[1]
    labs=[]; vals=[]
    for nm,rec in RB.items():
        v=[rec['ref_poav']['piIDT1_totpers']]+[d['piIDT1_totpers'] for d in rec['rot']]
        vals.append(np.array(v)/v[0]); labs.append(nm)
    ax.boxplot(vals,labels=[l[:6] for l in labs],widths=.5)
    ax.axhline(1,color='k',lw=.6,ls='--')
    ax.set_ylabel(r'$T_1/T_1^{\rm ref}$'); ax.set_title('(b) 6 random rotations',fontsize=7.5)
    ax.tick_params(axis='x',labelsize=6)
    ax=axs[2]
    RBx={k:v for k,v in RB.items() if 'xc' in v}
    fx=list(RBx[list(RBx)[0]]['xc'].keys()) if RBx else []
    w=0.8/max(len(RBx),1)
    for i,(nm,rec) in enumerate(RBx.items()):
        ref=rec['xc']['wb97x-d3bj']['piIDT1_totpers']
        ax.bar(np.arange(len(fx))+i*w-0.4+w/2,
               [rec['xc'][f]['piIDT1_totpers']/ref for f in fx],width=w,label=nm)
    ax.set_xticks(range(len(fx))); ax.set_xticklabels(fx,rotation=25,ha='right',fontsize=6)
    ax.axhline(1,color='k',lw=.6,ls='--'); ax.set_ylabel(r'$T_1$ (rel.)')
    ax.set_ylim(0.90,1.10)
    ax.set_title('(c) functional sensitivity',fontsize=7.5)
    if RBx: ax.legend(fontsize=5.5)
    fig.savefig(f'{P}/figs/fig3.pdf'); plt.close(fig)

# --------------------------------------------------------------- regression
if os.path.exists(f'{R}/regression.json'):
    G=json.load(open(f'{R}/regression.json'))
    PL={'eint_Na+':r'Na$^+$ (cation--$\pi$)','eint_Cl-':r'Cl$^-$ (anion--$\pi$)',
        'eint_benzene':r'benzene ($\pi$--$\pi$)','eint_CH4':r'CH$_4$ (CH--$\pi$)'}
    SETS=['baseline','baseline+Npi','piIDT only','baseline+piIDT']
    HDR={'baseline':'baseline','baseline+Npi':r'\,+\,$N_\pi$',
         'piIDT only':r'$\pi$-IDT only','baseline+piIDT':r'baseline\,+\,$\pi$-IDT'}
    lines=[]
    for t in PL:
        if t not in G: continue
        g=G[t]
        cells=[]
        for s_ in SETS:
            m=g[s_]
            pp = m.get('p_perm')
            cells.append('$%.2f$ [$%.2f$, $%.2f$]'%(m['q2_mean'],m['q2_lo'],m['q2_hi']))
            cells.append('n/a' if pp is None else ('$<$0.01' if pp<=0.005 else '%.2f'%pp))
        lines.append('%s & %d & %.2f & %s'%(PL[t],g['n'],g['y_sd'],' & '.join(cells)))
    tex=(r'\begin{tabular}{lcc' + 'cc'*len(SETS) + r'}' + '\n\\toprule\n'
         + r'& & & ' + ' & '.join(r'\multicolumn{2}{c}{%s}'%HDR[s_] for s_ in SETS) + r'\\'
         + '\n' + ' '.join(r'\cmidrule(lr){%d-%d}'%(4+2*i,5+2*i) for i in range(len(SETS)))
         + '\n' + r'probe & $n$ & s.d. & ' + ' & '.join([r'$Q^2$ & $p$']*len(SETS)) + r'\\'
         + '\n\\midrule\n' + '\\\\\n'.join(lines) + '\\\\\n\\bottomrule\n\\end{tabular}')
    open(f'{P}/tables/table_reg.tex','w').write(tex)

    # univariate correlation table
    UV=pd.read_csv(f'{R}/univariate.csv').set_index('feature')
    def fmt(v):
        x=float(v); x=0.0 if abs(x)<0.005 else x
        return '$%+0.2f$'%x
    GRP=[('molecular size',['nheavy','vdw_area','mw']),
         ('electrostatic',['esp_min','esp_max','esp_var','esp_axial_2.6']),
         (r'$\pi$ density',['Npi_grid','Api_integral','occlusion_ratio','A_pi_max']),
         (r'$\pi$-IDT topology',['piIDT0_n','piIDT0_totpers','piIDT0_entropy',
                                 'piIDT1_n','piIDT1_totpers','piIDT1_maxpers','piIDT1_entropy'])]
    NAME={'nheavy':r'$N_{\rm heavy}$','vdw_area':r'vdW area','mw':r'$M_{\rm w}$',
          'esp_min':r'$V_{\min}$','esp_max':r'$V_{\max}$','esp_var':r'$\sigma^2_V$',
          'esp_axial_2.6':r'$V_{\rm ax}(2.6)$','Npi_grid':r'$N_\pi$',
          'Api_integral':r'$I_\pi$','occlusion_ratio':r'$\Omega$','A_pi_max':r'$\max A_\pi$',
          'piIDT0_n':r'$\beta_0$','piIDT0_totpers':r'$T_0$','piIDT0_entropy':r'$E_0$',
          'piIDT1_n':r'$\beta_1$','piIDT1_totpers':r'$T_1$','piIDT1_maxpers':r'$\ell_1^{\max}$',
          'piIDT1_entropy':r'$E_1$'}
    ls=[]
    for gname,feats in GRP:
        ls.append(r'\multicolumn{5}{l}{\emph{%s}}'%gname)
        for f in feats:
            if f not in UV.index: continue
            ls.append('\\quad %s & %s & %s & %s & %s'%(NAME.get(f,f),
                 fmt(UV.loc[f,'eint_Na+']),fmt(UV.loc[f,'eint_Cl-']),
                 fmt(UV.loc[f,'eint_benzene']),fmt(UV.loc[f,'eint_CH4'])))
    tex=(r'\begin{tabular}{lcccc}' + '\n\\toprule\n'
         + r'descriptor & Na$^+$ & Cl$^-$ & benzene & CH$_4$\\' + '\n\\midrule\n'
         + '\\\\\n'.join(ls) + '\\\\\n\\bottomrule\n\\end{tabular}')
    open(f'{P}/tables/table_uni.tex','w').write(tex)

    keys=[t for t in PL if t in G]
    fig,axs=plt.subplots(1,len(keys),figsize=(7.0,2.0))
    axs=np.atleast_1d(axs)
    for ax,t in zip(axs,keys):
        g=G[t]; y=np.array(g['y'])
        for s_,c,mk in [('baseline','#8d99ae','o'),('baseline+piIDT','#c1121f','s')]:
            p_=np.array(g[s_]['pred'])
            ax.scatter(y,p_,s=14,c=c,marker=mk,alpha=.85,linewidths=0,
                       label='%s ($Q^2$=%.2f)'%('baseline' if s_=='baseline' else r'+$\pi$-IDT',
                                                g[s_]['q2_mean']))
        lo=min(y.min(),np.array(g['baseline']['pred']).min())
        hi=max(y.max(),np.array(g['baseline']['pred']).max())
        ax.plot([lo,hi],[lo,hi],'k-',lw=.6)
        ax.set_title(PL[t],fontsize=7); ax.set_xlabel(r'$E_{\rm int}$ / kcal mol$^{-1}$')
        ax.legend(fontsize=5.2,loc='upper left')
    axs[0].set_ylabel('cross-validated prediction')
    for ax in axs: ax.tick_params(labelsize=6)
    fig.savefig(f'{P}/figs/fig4.pdf'); plt.close(fig)
    for t in PL:
        if t not in G: continue
        tag=t.split('_')[1].replace('+','P').replace('-','M').replace('4','four')
        N('qtwobase'+tag,'%.2f'%G[t]['baseline']['q2_mean'])
        N('qtwofull'+tag,'%.2f'%G[t]['baseline+piIDT']['q2_mean'])
        N('qtwoidt'+tag,'%.2f'%G[t]['piIDT only']['q2_mean'])
        N('pgain'+tag,'%.2f'%G[t]['boot_gain']['p_gt0'])
        N('nreg'+tag, G[t]['n'])

if os.path.exists(f'{R}/rotation_v2.json'):
    RV=json.load(open(f'{R}/rotation_v2.json'))
    def rng_(vals):
        v=np.array(vals,float)
        return (v.max()-v.min())/abs(v.mean()) if abs(v.mean())>1e-12 else 0.0
    ip=[]; cn=[]
    for nm,rec in RV.items():
        for k in ['piIDT1_n','piIDT1_totpers','piIDT0_n','Api_integral']:
            ip.append(rng_([d[k] for d in rec['inplane']]))
            cn.append(rng_([d[k] for d in rec['canon']]))
    N('canonMaxRel',sci(max(cn),2))
    ip2=[]
    for nm,rec in RV.items():
        for k in ['piIDT0_n','piIDT0_totpers','piIDT0_maxpers','piIDT0_entropy',
                  'piIDT1_n','piIDT1_totpers','piIDT1_maxpers','piIDT1_entropy',
                  'Api_integral']:
            ip2.append(rng_([d[k] for d in rec['inplane']]))
    N('inplaneMaxRel','%.3f'%max(ip2))
    ipb=[]
    for nm,rec in RV.items():
        for k in ['piIDT0_n','piIDT1_n']:
            ipb.append(rng_([d[k] for d in rec['inplane']]))
    N('inplaneCountRel','%.0f'%max(ipb))

if os.path.exists(f'{R}/shield_poav.json'):
    SP=json.load(open(f'{R}/shield_poav.json'))
    ser=['benzene','toluene','p-xylene','mesitylene','hexamethylbenzene','hexaethylbenzene']
    N('shieldI',' & '.join('%.3f'%SP[m]['I'] for m in ser))
    N('shieldOm',' & '.join('%.3f'%SP[m]['Om'] for m in ser))
    N('shieldNpiRing',', '.join('%.2f'%SP[m]['Npi_ring'] for m in ser))
    N('shieldDrop','%.0f'%(100*(1-SP[ser[-1]]['I']/SP[ser[0]]['I'])))

if os.path.exists(f'{R}/rigid_vs_relaxed.json'):
    RR=json.load(open(f'{R}/rigid_vs_relaxed.json'))
    N('rigidRelaxNaR','%.4f'%RR['Na+']['r']); N('rigidRelaxClR','%.4f'%RR['Cl-']['r'])
    N('rigidRelaxN',RR['Na+']['n'])
    N('rigidRelaxShiftNa','%.2f'%abs(RR['Na+']['mean_shift']))
    N('rigidRelaxShiftCl','%.2f'%abs(RR['Cl-']['mean_shift']))

if os.path.exists(f'{R}/threshold_rel.json'):
    TR=json.load(open(f'{R}/threshold_rel.json'))
    N('relThreshBest',max(TR.values())); N('relThreshWorst',min(TR.values()))

if os.path.exists(f'{R}/mask_series.json'):
    MS=json.load(open(f'{R}/mask_series.json'))
    SER=['benzene','toluene','p-xylene','mesitylene','hexamethylbenzene','hexaethylbenzene']
    SER=[m for m in SER if m in MS]
    keys=sorted(MS[SER[0]].keys())
    from scipy import stats as _st
    mono=0; taus=[]
    for k in keys:
        I=[MS[m][k]['I'] for m in SER]
        tau=_st.kendalltau(range(len(SER)),I).statistic
        taus.append(tau)
        if all(I[i]>I[i+1] for i in range(len(I)-1)): mono+=1
    N('maskSettings',len(keys)); N('maskMono',mono)
    fac=[max(MS[m][k]['I'] for k in keys)/min(MS[m][k]['I'] for k in keys) for m in SER]
    N('maskFactorMin','%.0f'%min(fac)); N('maskFactorMax','%.0f'%max(fac))
    N('maskTauWorst','%.2f'%max(taus))
    rows=[]
    for k in keys:
        sc,dl=k.split('_')
        rows.append('%s & %s & %s & %s'%(sc,dl,
            ' & '.join('%.2f'%MS[m][k]['I'] for m in SER),
            'yes' if all(MS[SER[i]][k]['I']>MS[SER[i+1]][k]['I'] for i in range(len(SER)-1)) else 'no'))
    tex=(r'\begin{tabular}{cc'+'c'*len(SER)+r'c}'+'\n\\toprule\n'
         +r'$\lambda$ & $\delta$/\AA & '+' & '.join(m.replace('hexamethylbenzene','HMB').replace('hexaethylbenzene','HEB').replace('p-xylene','$p$-xylene') for m in SER)+r' & monotonic\\'
         +'\n\\midrule\n'+'\\\\\n'.join(rows)+'\\\\\n\\bottomrule\n\\end{tabular}')
    open(f'{P}/tables/table_mask.tex','w').write(tex)

# ------------------------------------------------------------------ timings
df=pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(list,dict))}
                 for r in A.values()]).set_index('name')
N('tScfMed','%.1f'%df['t_scf'].median()); N('tGridMed','%.1f'%df['t_grid'].median())
N('tTotMax','%.0f'%df['t_total'].max()); N('tTotMed','%.0f'%df['t_total'].median())
N('nMol',len(df)); N('gridMax','%.1f'%(df['ngrid'].max()/1e6))

with open(f'{P}/numbers.tex','w') as f:
    for k,v in NUM.items():
        f.write('\\newcommand{\\%s}{%s}\n'%(''.join(c for c in k if c.isalpha()), v))
json.dump(NUM, open(f'{R}/numbers.json','w'), indent=1)
print(json.dumps(NUM, indent=1))

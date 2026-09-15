import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import sys, os, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':8,'axes.linewidth':0.6,'xtick.major.width':0.6,
                     'ytick.major.width':0.6,'legend.frameon':False,
                     'font.family':'serif','mathtext.fontset':'dejavuserif',
                     'savefig.bbox':'tight','savefig.pad_inches':0.02})
R=str(paths.RESULTS); P=str(paths.PAPER)
A=json.load(open(f'{R}/setA.json'))
NUM={}
def num(k,v,fmt='{:.3f}'):
    NUM[k]= fmt.format(v) if isinstance(v,float) else str(v)
    return v

from rdkit import Chem
def nring(r):
    rd = Chem.AddHs(Chem.MolFromSmiles(r['smiles'])); pia = set(r['pi_atoms'])
    return sum(1 for ring in rd.GetRingInfo().AtomRings() if all(a in pia for a in ring))
df = pd.DataFrame([{k:v for k,v in r.items()
                    if not isinstance(v,(list,dict))} for r in A.values()])
df = df.set_index('name')
df['n_pi_rings'] = pd.Series({n: nring(r) for n, r in A.items()})
df.to_csv(f'{R}/setA_table.csv')

# ---------------------------------------------------------------- Table 1
ORDER = ['ethene','butadiene','hexatriene','benzene','naphthalene','anthracene',
 'tetracene','pentacene','phenanthrene','pyrene','triphenylene','perylene',
 'coronene','benzo[a]pyrene','azulene','biphenyl','pyridine','pyrimidine',
 's-triazine','pyrrole','furan','thiophene','imidazole','indole','quinoline',
 'porphine','toluene','p-xylene','mesitylene','hexamethylbenzene',
 'hexaethylbenzene','1,3,5-tri-tBu-benzene','fluorobenzene','hexafluorobenzene',
 'phenol','aniline','nitrobenzene','benzonitrile','corannulene']
ORDER=[m for m in ORDER if m in df.index]

def esc(s): return s.replace('_','\\_').replace('[','{[}').replace(']','{]}')

def table1():
    rows=[]
    for m in ORDER:
        r=df.loc[m]
        npi = r['Npi_sym'] if r['partition']=='symmetry' else r.get('Npi_poav', float('nan'))
        rows.append('%s & %d & %s & %.2f & %d & %.3f & %d & %d & %d & %.3f & %.3f'%(
            esc(m), r['nheavy'], r['partition'][0].upper(), npi,
            r['n_pi_rings'], r['occlusion_ratio'],
            r['piIDT0_n'], r['piIDT1_n'], r['piIDT2_n'],
            r['piIDT1_totpers'], r['Api_integral']))
    body='\\\\\n'.join(rows)
    tex = (r"""\begin{tabular}{lcccccccccc}
\toprule
Molecule & $N_{\rm heavy}$ & P & $N_\pi$ & $N_{\rm ring}$ & $\Omega$ &
$\beta_0$ & $\beta_1$ & $\beta_2$ & $T_1$ & $I_\pi$\\
\midrule
""" + body + r"""\\
\bottomrule
\end{tabular}""")
    open(f'{P}/tables/table1.tex','w').write(tex)
table1()

# ------------------------------------------------- acene / ring-count result
ring_sets = [m for m in ORDER if df.loc[m,'n_pi_rings']>0]
okall = sum(1 for m in ORDER if df.loc[m,'piIDT1_n']==2*df.loc[m,'n_pi_rings'])
ok = sum(1 for m in ring_sets if df.loc[m,'piIDT1_n']==2*df.loc[m,'n_pi_rings'])
num('nRingSystems', len(ring_sets)); num('nRingMatch', ok); num('nRingMatchB', okall)
plan=[m for m in ORDER if df.loc[m,'partition']=='symmetry']
num('nPlanarMatch', sum(1 for m in plan if df.loc[m,'piIDT1_n']==2*df.loc[m,'n_pi_rings']))
num('nPlanar', len(plan))
mism = [m for m in ORDER if df.loc[m,'piIDT1_n']!=2*df.loc[m,'n_pi_rings']]
json.dump({'match':ok,'total':len(ring_sets),'mismatch':
           {m:[int(df.loc[m,'piIDT1_n']),int(df.loc[m,'n_aromatic_rings'])] for m in mism}},
          open(f'{R}/ringcount.json','w'), indent=1)

# ------------------------------------------------------------------- Fig 2
fig,axs=plt.subplots(1,2,figsize=(6.8,2.6))
ax=axs[0]
grp = {'acene':'o','PAH':'s','heteroarene':'^','macrocycle':'D',
       'shielding':'v','substituted':'<','curved':'*','biaryl':'p','reference':'x'}
for g,mk in grp.items():
    sel=[m for m in ORDER if df.loc[m,'group']==g]
    if not sel: continue
    ax.scatter(df.loc[sel,'n_pi_rings'], df.loc[sel,'piIDT1_n'],
               marker=mk, s=22, label=g, alpha=.85, linewidths=.6)
lim=[-0.4, df.loc[ORDER,'n_pi_rings'].max()+0.6]
ax.plot(lim,[2*l for l in lim],'k--',lw=.7,label=r'$\beta_1=2N_{\rm ring}$')
ax.set_xlabel(r'number of $\pi$ rings $N_{\rm ring}$')
ax.set_ylabel(r'$\pi$-IDT$_1$ loop count $\beta_1$')
ax.set_xlim(*lim); ax.legend(fontsize=6, ncol=2, loc='upper left')

ax=axs[1]
sh=['benzene','toluene','p-xylene','mesitylene','hexamethylbenzene','hexaethylbenzene']
sh=[m for m in sh if m in df.index]
nm=[0,1,2,3,6,6]
import os as _os
if _os.path.exists(f'{R}/shield_poav.json'):
    SP=json.load(open(f'{R}/shield_poav.json'))
    yI=[SP[m]['I'] for m in sh]; yO=[SP[m]['Om'] for m in sh]
else:
    yI=list(df.loc[sh,'Api_integral']); yO=list(df.loc[sh,'occlusion_ratio'])
ax.plot(range(len(sh)), yI,'o-',lw=1,ms=4,color='#1b4965',
        label=r'$I_\pi$ (accessible)')
ax2=ax.twinx()
ax2.plot(range(len(sh)), yO,'s--',lw=1,ms=4,color='#c1121f',
         label=r'$\Omega$ (occlusion)')
ax.set_xticks(range(len(sh)))
ax.set_xticklabels(['benzene','tol.','$p$-xyl.','mesit.','HMB','HEB'],rotation=35,ha='right')
ax.set_ylabel(r'$I_\pi$ / e'); ax2.set_ylabel(r'$\Omega$')
ax.set_title(r'constant $N_{\rm ring}=1$, constant formal $N_\pi=6$',fontsize=7)
h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
ax.legend(h1+h2,l1+l2,fontsize=6,loc='center left')
fig.savefig(f'{P}/figs/fig2.pdf')
plt.close(fig)

json.dump(NUM, open(f'{R}/numbers_partial.json','w'), indent=1)
print('table1 + fig2 written; ring match %d/%d (all %d/%d); mismatches %s'%(ok,len(ring_sets),okall,len(ORDER),mism))

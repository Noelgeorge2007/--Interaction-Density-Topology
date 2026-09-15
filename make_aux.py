"""Generate the auxiliary result files consumed by the analysis scripts:
nring.json, threshold_sens.json and univariate.csv."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import sys, json
import numpy as np, pandas as pd
from rdkit import Chem
from scipy import stats
R=str(paths.RESULTS)
A=json.load(open(f'{R}/setA.json'))

def nring(r):
    rd=Chem.AddHs(Chem.MolFromSmiles(r['smiles'])); pia=set(r['pi_atoms'])
    return sum(1 for ring in rd.GetRingInfo().AtomRings() if all(a in pia for a in ring))
K={n:nring(r) for n,r in A.items()}
json.dump(K, open(f'{R}/nring.json','w'), indent=1)

ths=[1e-3,2e-3,3e-3,4e-3,5e-3,6e-3,8e-3,1e-2,1.2e-2]
out={}
for th in ths:
    c=0
    for n,r in A.items():
        b=np.array(r['bars']['1']).reshape(-1,2)
        L=(b[:,0]-b[:,1]) if len(b) else np.array([])
        if int((L>th).sum())==2*K[n]: c+=1
    out[str(th)]=c
json.dump(out, open(f'{R}/threshold_sens.json','w'), indent=1)
print('nring + threshold_sens written:', out)

D=pd.read_csv(f'{R}/setB_table.csv').set_index('name')
T=['eint_Na+','eint_Cl-','eint_benzene','eint_CH4']
FEAT=['nheavy','vdw_area','mw','esp_min','esp_max','esp_var','esp_axial_2.6',
      'Npi_grid','Api_integral','occlusion_ratio','A_pi_max',
      'piIDT0_n','piIDT0_totpers','piIDT1_n','piIDT1_totpers','piIDT1_maxpers',
      'piIDT0_entropy','piIDT1_entropy']
rows=[]
for f in FEAT:
    if f not in D.columns: continue
    r={'feature':f}
    for t in T:
        x=D[f].values.astype(float); y=D[t].values.astype(float)
        m=np.isfinite(x)&np.isfinite(y)
        r[t]='%+.2f'%stats.pearsonr(x[m],y[m])[0]
        r[t+'_s']='%+.2f'%stats.spearmanr(x[m],y[m])[0]
    rows.append(r)
pd.DataFrame(rows).set_index('feature').to_csv(f'{R}/univariate.csv')
print('univariate.csv written')

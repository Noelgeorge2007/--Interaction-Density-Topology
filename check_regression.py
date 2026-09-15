import ast,json,sys
from pathlib import Path
import numpy as np
from scipy import stats
r=Path(__file__).resolve().parents[1];tree=ast.parse((r/'src/regress2.py').read_text())
selected=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('fit_predict','repeated_cv')],type_ignores=[])
ns=dict(np=np,stats=stats,NREP=50,KFOLD=5,ALPHAS=np.logspace(-2,5,60))
exec(compile(selected,'regress2-functions','exec'),ns)
A=json.loads((r/'results/setA.json').read_text())
for n,v in json.loads((r/'results/esp_axial.json').read_text()).items():
 if n in A:A[n].update(v)
G=json.loads((r/'results/regression.json').read_text());out=[]
for t,g in G.items():
 for name,model in g.items():
  if not isinstance(model,dict) or 'features' not in model:continue
  X=np.array([[A[n][c] for c in model['features']] for n in g['names']]);y=np.array(g['y'])
  new=ns['repeated_cv'](X,y)
  delta=abs(new['q2_mean']-model['q2_mean'])
  assert np.allclose(new['pred'],model['pred'],rtol=1e-7,atol=1e-7),(t,name,'pred')
  assert delta<1e-7,(t,name,delta)
  out.append(dict(target=t,model=name,q2_mean=new['q2_mean'],delta=delta))
# Refits intentionally exclude permutation tests and bootstrap confidence limits.
print('PASS',len(out),'model refits; max Q2 delta',max(o['delta'] for o in out))

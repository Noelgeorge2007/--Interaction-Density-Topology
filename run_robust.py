"""Invariance, convergence and sensitivity tests for pi-IDT.
SCF results are reused across all tests that do not change the wavefunction."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import sys, os, json, time
import numpy as np, pidt, opt
from rdkit import Chem

OUT=str(paths.RESULTS / 'robust.json')
res = json.load(open(OUT)) if os.path.exists(OUT) else {}
FULL = [('benzene','c1ccccc1'), ('pyridine','c1ccncc1'),
        ('naphthalene','c1ccc2ccccc2c1')]
LIGHT= [('coronene','c1cc2ccc3ccc4ccc5ccc6ccc1c1c2c3c4c5c61')]

def field(mol, dmp, syms, xyz, pia, h, pad, delta, scale):
    G, shape, _ = pidt.make_grid(xyz, pad=pad, h=h)
    A = pidt.eval_density(mol, dmp, G) * pidt.accessibility_mask(
            G, syms, xyz, pia, delta=delta, scale=scale)
    d = pidt.descriptors(pidt.persistence(A, shape))
    d['Api_integral'] = float(A.sum()*(h/pidt.BOHR)**3)
    return d

def wf(syms, xyz, basis='6-31G*', xc='wb97x-d3bj', force_poav=False, rd=None):
    mol, mf = pidt.run_dft(syms, xyz, basis=basis, xc=xc)
    dm,_ = pidt.valence_dm(mf)
    planar = np.abs(np.array(xyz)[:,2]).max() < 0.05
    if planar and not force_poav:
        dmp,_,_ = pidt.pi_density_matrix_symmetry(mol, dm)
    else:
        dmp,_ = pidt.pi_density_matrix_poav(mol, dm, syms, xyz, rd)
    pia,_ = pidt.pi_atom_set(mol, dmp)
    return mol, dmp, pia

def rand_rot(seed):
    rng=np.random.default_rng(seed); q=rng.normal(size=4); q/=np.linalg.norm(q)
    w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])

def run(name, smi, light=False):
    t0=time.time()
    s,x0,rd = pidt.geometry_from_smiles(smi)
    g = opt.prepare_geometry(s,x0); x=g['xyz']
    rec={'name':name}
    mol,dmp,pia = wf(s,x,rd=rd)
    rec['ref'] = field(mol,dmp,s,x,pia,0.15,3.5,1.0,1.0)
    rec['grid']= {str(h): field(mol,dmp,s,x,pia,h,3.5,1.0,1.0)
                  for h in (0.30,0.25,0.20,0.15,0.12)}
    rec['pad'] = {str(p): field(mol,dmp,s,x,pia,0.15,p,1.0,1.0) for p in (2.5,3.5,4.5)}
    rec['mask']= {f'{sc}_{dl}': field(mol,dmp,s,x,pia,0.15,3.5,dl,sc)
                  for sc in (0.9,1.0,1.1) for dl in (0.5,1.0,1.5)}
    m2,d2,p2 = wf(s,x,force_poav=True,rd=rd)
    rec['ref_poav'] = field(m2,d2,s,x,p2,0.15,3.5,1.0,1.0)
    nrot = 3 if light else 6
    rec['rot']=[]
    for k in range(nrot):
        xr = x @ rand_rot(k).T
        mr,dr,pr = wf(s,xr,force_poav=True,rd=rd)
        rec['rot'].append(field(mr,dr,s,xr,pr,0.15,3.5,1.0,1.0))
    if not light:
        rec['basis']={}
        for b in ('6-31G*','6-311G**','def2-TZVP'):
            mb,db,pb = wf(s,x,basis=b,rd=rd)
            rec['basis'][b]=field(mb,db,s,x,pb,0.15,3.5,1.0,1.0)
        rec['xc']={}
        for f in ('wb97x-d3bj','b3lyp','pbe0','pbe'):
            mf_,df_,pf_ = wf(s,x,xc=f,rd=rd)
            rec['xc'][f]=field(mf_,df_,s,x,pf_,0.15,3.5,1.0,1.0)
        rec['jitter']={}
        for j in (0.01,0.03,0.05):
            lst=[]
            for k in range(3):
                xj = x + np.random.default_rng(100*k+int(j*1000)).normal(scale=j,size=x.shape)
                mj,dj,pj = wf(s,xj,force_poav=True,rd=rd)
                lst.append(field(mj,dj,s,xj,pj,0.15,3.5,1.0,1.0))
            rec['jitter'][str(j)]=lst
    rec['t']=time.time()-t0
    return rec

for name,smi in FULL+LIGHT:
    if name in res: print('skip',name,flush=True); continue
    rec = run(name, smi, light=(name,smi) in LIGHT)
    res[name]=rec; json.dump(res,open(OUT,'w'),indent=1)
    print('%s done in %.0f s'%(name,rec['t']), flush=True)

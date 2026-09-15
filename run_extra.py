import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import sys, os, json, time
import numpy as np, pidt
from rdkit import Chem
R=str(paths.RESULTS)
A=json.load(open(f'{R}/setA.json'))

# ---- (1) POAV vs exact partition, with the corrected lone-pair rule ---------
out={}
f=f'{R}/poav_v2.json'
if os.path.exists(f): out=json.load(open(f))
for n,r in A.items():
    if r['partition']!='symmetry' or n in out: continue
    syms=r['symbols']; xyz=np.array(r['coords'])
    rd=Chem.AddHs(Chem.MolFromSmiles(r['smiles']))
    mol,mf=pidt.run_dft(syms,xyz,basis=r['basis'],xc=r['xc'])
    dm,_=pidt.valence_dm(mf); S=mol.intor('int1e_ovlp')
    ds,_,leak=pidt.pi_density_matrix_symmetry(mol,dm)
    dp,npi=pidt.pi_density_matrix_poav(mol,dm,syms,xyz,rd)
    out[n]=dict(Npi_sym=float(np.einsum('ij,ji->',ds,S)), Npi_poav=float(npi),
                reldiff=float(np.linalg.norm(ds-dp)/np.linalg.norm(ds)), leak=float(leak))
    json.dump(out,open(f,'w'),indent=1)
    print('%-22s Npi_sym=%7.3f Npi_poav=%7.3f reldiff=%.3f'%(n,out[n]['Npi_sym'],npi,out[n]['reldiff']),flush=True)

# ---- (2) in-plane rotation and canonical-frame invariance -------------------
out2={}
f2=f'{R}/rotation_v2.json'
if os.path.exists(f2): out2=json.load(open(f2))
def desc_of(syms,xyz,rd,canon=False):
    x=np.array(xyz)
    mol,mf=pidt.run_dft(syms,x,basis='6-31G*',xc='wb97x-d3bj')
    dm,_=pidt.valence_dm(mf)
    dmp,_=pidt.pi_density_matrix_poav(mol,dm,syms,x,rd)
    pia,_=pidt.pi_atom_set(mol,dmp)
    xg = pidt.canonical_frame(syms,x,pia) if canon else x
    if canon:
        mol,mf=pidt.run_dft(syms,xg,basis='6-31G*',xc='wb97x-d3bj')
        dm,_=pidt.valence_dm(mf)
        dmp,_=pidt.pi_density_matrix_poav(mol,dm,syms,xg,rd)
        pia,_=pidt.pi_atom_set(mol,dmp)
    G,shape,_=pidt.make_grid(xg,pad=3.5,h=0.15)
    Af=pidt.eval_density(mol,dmp,G)*pidt.accessibility_mask(G,syms,xg,pia)
    d=pidt.descriptors(pidt.persistence(Af,shape))
    d['Api_integral']=float(Af.sum()*(0.15/pidt.BOHR)**3)
    return d
def rotz(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.0]])
def rand_rot(seed):
    rng=np.random.default_rng(seed); q=rng.normal(size=4); q/=np.linalg.norm(q); w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
for n in ['benzene','pyridine','naphthalene']:
    if n in out2: continue
    r=A[n]; syms=r['symbols']; xyz=np.array(r['coords'])
    rd=Chem.AddHs(Chem.MolFromSmiles(r['smiles']))
    rec={'inplane':[desc_of(syms,xyz@rotz(a).T,rd) for a in np.linspace(0,np.pi/3,5)],
         'canon':[desc_of(syms,xyz@rand_rot(k).T,rd,canon=True) for k in range(4)]}
    out2[n]=rec; json.dump(out2,open(f2,'w'),indent=1)
    print('%s rotation tests done'%n,flush=True)

# ---- (3) mask-parameter robustness of the shielding series ------------------
out3={}
f3=f'{R}/mask_series.json'
if os.path.exists(f3): out3=json.load(open(f3))
SER=['benzene','toluene','p-xylene','mesitylene','hexamethylbenzene','hexaethylbenzene']
for n in SER:
    if n in out3: continue
    r=A[n]; syms=r['symbols']; xyz=np.array(r['coords'])
    rd=Chem.AddHs(Chem.MolFromSmiles(r['smiles']))
    mol,mf=pidt.run_dft(syms,xyz,basis=r['basis'],xc=r['xc'])
    dm,_=pidt.valence_dm(mf)
    if r['partition']=='symmetry': dmp,_,_=pidt.pi_density_matrix_symmetry(mol,dm)
    else: dmp,_=pidt.pi_density_matrix_poav(mol,dm,syms,xyz,rd)
    pia,_=pidt.pi_atom_set(mol,dmp)
    G,shape,_=pidt.make_grid(xyz,pad=3.5,h=0.15)
    rho=pidt.eval_density(mol,dmp,G); dV=(0.15/pidt.BOHR)**3
    rec={}
    for sc in (0.9,1.0,1.1):
        for dl in (0.5,1.0,1.5):
            Af=rho*pidt.accessibility_mask(G,syms,xyz,pia,delta=dl,scale=sc)
            rec['%.1f_%.1f'%(sc,dl)]=dict(I=float(Af.sum()*dV),
                Om=float(1-Af.sum()/rho.sum()),
                b1=pidt.descriptors(pidt.persistence(Af,shape))['piIDT1_n'])
    out3[n]=rec; json.dump(out3,open(f3,'w'),indent=1)
    print('%-20s mask sweep done'%n,flush=True)
print('EXTRA DONE')

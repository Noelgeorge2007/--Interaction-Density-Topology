"""Axial ESP above the pi face: the standard electrostatic descriptor for
cation-pi/anion-pi propensity, computed at the same site as the probes."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import sys, os, json
import numpy as np, pidt, probes
from rdkit import Chem
OUT=str(paths.RESULTS / 'esp_axial.json')
A=json.load(open(str(paths.RESULTS / 'setA.json')))
res=json.load(open(OUT)) if os.path.exists(OUT) else {}
for n,r in A.items():
    if n in res: continue
    syms=r['symbols']; xyz=np.array(r['coords'])
    rd=Chem.AddHs(Chem.MolFromSmiles(r['smiles']))
    mol,mf=pidt.run_dft(syms,xyz,basis=r['basis'],xc=r['xc'])
    dm=mf.make_rdm1()
    c,nv=probes.probe_site(syms,xyz,rd)
    out={}
    for d in (2.0,2.6,3.2):
        p=((c+d*nv)/pidt.BOHR).reshape(1,3)
        V=-np.einsum('pij,ji->p',mol.intor('int1e_grids',grids=p),dm)[0]
        V+=sum(z/np.linalg.norm(p[0]-R) for z,R in zip(mol.atom_charges(),mol.atom_coords()))
        out['esp_axial_%.1f'%d]=float(V)
    res[n]=out; json.dump(res,open(OUT,'w'),indent=1)
    print('%-22s'%n, {k:round(v,5) for k,v in out.items()}, flush=True)

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import sys, os, json
import numpy as np, pidt
from rdkit import Chem
R=str(paths.RESULTS)
A=json.load(open(f'{R}/setA.json')); K=json.load(open(f'{R}/nring.json'))

# (1) relative persistence thresholds, saved
rel={}
for f in (0.02,0.05,0.08,0.10,0.15,0.20):
    c=0
    for n,r in A.items():
        b=np.array(r['bars']['1']).reshape(-1,2)
        L=(b[:,0]-b[:,1]) if len(b) else np.array([])
        if int((L> f*r['A_pi_max']).sum())==2*K[n]: c+=1
    rel['%.2f'%f]=c
json.dump(rel, open(f'{R}/threshold_rel.json','w'), indent=1)
print('relative thresholds:', rel, flush=True)

# (2) where the pi density sits around the heteroatom
loc={}
for n in ['thiophene','furan','pyrrole','imidazole','benzene']:
    r=A[n]; syms=r['symbols']; xyz=np.array(r['coords'])
    rd=Chem.AddHs(Chem.MolFromSmiles(r['smiles']))
    mol,mf=pidt.run_dft(syms,xyz,basis=r['basis'],xc=r['xc'])
    dm,_=pidt.valence_dm(mf); dmp,_,_=pidt.pi_density_matrix_symmetry(mol,dm)
    pia,pop=pidt.pi_atom_set(mol,dmp)
    G,shape,_=pidt.make_grid(xyz,pad=3.5,h=0.15)
    rho=pidt.eval_density(mol,dmp,G); S=pidt.accessibility_mask(G,syms,xyz,pia)
    Af=rho*S; dV=(0.15/pidt.BOHR)**3
    het=[i for i,s in enumerate(syms) if s in ('S','O','N')]
    rec=dict(I=float(Af.sum()*dV), rho_max=float(rho.max()))
    for i in het:
        d=np.linalg.norm(G-xyz[i],axis=1)
        rec['frac_within_1.5_atom%d_%s'%(i,syms[i])]=float(Af[d<1.5].sum()/Af.sum())
        rec['vol_above_5e-3_atom%d'%i]=float((Af[d<2.0]>5e-3).sum()*0.15**3)
    rec['vol_above_5e-3_total']=float((Af>5e-3).sum()*0.15**3)
    loc[n]=rec; print(n, {k:round(v,4) for k,v in rec.items()}, flush=True)
json.dump(loc, open(f'{R}/heteroatom_lobe.json','w'), indent=1)

# (3) shielding series computed uniformly with the POAV route
SER=['benzene','toluene','p-xylene','mesitylene','hexamethylbenzene','hexaethylbenzene']
uni={}
for n in SER:
    r=A[n]; syms=r['symbols']; xyz=np.array(r['coords'])
    rd=Chem.AddHs(Chem.MolFromSmiles(r['smiles']))
    mol,mf=pidt.run_dft(syms,xyz,basis=r['basis'],xc=r['xc'])
    dm,_=pidt.valence_dm(mf)
    dmp,npi=pidt.pi_density_matrix_poav(mol,dm,syms,xyz,rd)
    pia,pop=pidt.pi_atom_set(mol,dmp)
    G,shape,_=pidt.make_grid(xyz,pad=3.5,h=0.15)
    rho=pidt.eval_density(mol,dmp,G); Af=rho*pidt.accessibility_mask(G,syms,xyz,pia)
    dV=(0.15/pidt.BOHR)**3
    d=pidt.descriptors(pidt.persistence(Af,shape))
    ringC=[i for i,a in enumerate(rd.GetAtoms()) if a.GetIsAromatic()]
    uni[n]=dict(Npi=float(npi), Npi_ring=float(sum(pop[i] for i in ringC)),
                I=float(Af.sum()*dV), Om=float(1-Af.sum()/rho.sum()),
                b0=d['piIDT0_n'], b1=d['piIDT1_n'], T1=d['piIDT1_totpers'])
    print('%-20s'%n, {k:(round(v,4) if isinstance(v,float) else v) for k,v in uni[n].items()}, flush=True)
json.dump(uni, open(f'{R}/shield_poav.json','w'), indent=1)

# (4) rigid vs relaxed probe energies
B=json.load(open(f'{R}/setB.json'))
from scipy import stats
out={}
for kind,key in [('Na+','scan_Na+'),('Cl-','scan_Cl-')]:
    rg=[];rx=[];nm=[]
    for n,r in B.items():
        if key+'_emin' in r and np.isfinite(r[key+'_emin']):
            rg.append(r['eint_'+kind]); rx.append(r[key+'_emin']); nm.append(n)
    if len(rg)>2:
        out[kind]=dict(n=len(rg), r=float(stats.pearsonr(rg,rx)[0]),
                       mean_shift=float(np.mean(np.array(rx)-np.array(rg))),
                       molecules=nm, rigid=rg, relaxed=rx,
                       dmin=[B[n][key+'_dmin'] for n in nm])
        print(kind,'n=%d r=%.4f mean shift=%.2f kcal/mol'%(out[kind]['n'],out[kind]['r'],out[kind]['mean_shift']), flush=True)
json.dump(out, open(f'{R}/rigid_vs_relaxed.json','w'), indent=1)
print('EXTRA2 DONE')

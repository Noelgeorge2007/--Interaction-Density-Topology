import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import time, json, traceback
import numpy as np, pidt, opt
from molecules import SET_A
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

OUT  = os.environ.get('PIDT_OUT',str(paths.RESULTS / 'setA.json'))
H    = float(os.environ.get('PIDT_H', 0.15))
PAD  = float(os.environ.get('PIDT_PAD', 3.5))
BASIS= os.environ.get('PIDT_BASIS','6-31G*')
XC   = os.environ.get('PIDT_XC','wb97x-d3bj')
res  = json.load(open(OUT)) if os.path.exists(OUT) else {}

def analyse(name, smi, group):
    t0=time.time()
    syms, xyz0, rd = pidt.geometry_from_smiles(smi)
    g = opt.prepare_geometry(syms, xyz0)
    xyz = g['xyz']
    mol, mf = pidt.run_dft(syms, xyz, basis=BASIS, xc=XC)
    dm_all = mf.make_rdm1(); dm, ncore = pidt.valence_dm(mf); tscf=time.time()-t0
    S = mol.intor('int1e_ovlp')
    rec = dict(name=name, smiles=smi, group=group, natoms=len(syms),
               nheavy=int(sum(1 for s in syms if s!='H')), nao=int(mol.nao),
               E=float(mf.e_tot), e_xtb=g['e_xtb'], xtb_converged=g['converged'],
               planarity_heavy=g['planarity_heavy'], planarity_all=g['planarity_all'],
               max_out_of_plane=g['max_out_of_plane'], snapped=g['snapped'],
               t_scf=tscf, basis=BASIS, xc=XC, h=H, pad=PAD,
               Nelec=float(np.einsum('ij,ji->',dm_all,S)), n_core_mo=int(ncore),
               Nelec_valence=float(np.einsum('ij,ji->',dm,S)),
               coords=np.round(xyz,5).tolist(), symbols=syms)
    planar = g['snapped']
    rec['partition'] = 'symmetry' if planar else 'POAV'
    dmp_sym=None
    if planar:
        dmp_sym, idx, leak = pidt.pi_density_matrix_symmetry(mol, dm)
        rec['Npi_sym']=float(np.einsum('ij,ji->',dmp_sym,S))
        rec['sigma_pi_leakage']=float(leak); rec['n_pi_ao']=int(len(idx))
    try:
        dmp_poav, npi_poav = pidt.pi_density_matrix_poav(mol, dm, syms, xyz, rd)
        rec['Npi_poav']=float(npi_poav)
        if dmp_sym is not None:
            rec['poav_dm_reldiff']=float(np.linalg.norm(dmp_sym-dmp_poav)/np.linalg.norm(dmp_sym))
    except Exception as e:
        dmp_poav=None; rec['poav_error']=str(e)
    dmp = dmp_sym if dmp_sym is not None else dmp_poav
    if dmp is None: raise RuntimeError('no pi partition available')

    pia, pipop = pidt.pi_atom_set(mol, dmp)
    rec['pi_atom_pop'] = np.round(pipop,4).tolist()
    rec['pi_atoms'] = sorted(pia)
    G, shape, ax = pidt.make_grid(xyz, pad=PAD, h=H)
    t1=time.time()
    rho_pi  = pidt.eval_density(mol, dmp, G)
    Sm      = pidt.accessibility_mask(G, syms, xyz, pia)
    A       = rho_pi*Sm
    dV      = (H/pidt.BOHR)**3
    rec.update(ngrid=int(len(G)), grid_shape=list(shape),
               Npi_grid=float(rho_pi.sum()*dV), Api_integral=float(A.sum()*dV),
               occlusion_ratio=float(1.0-A.sum()/max(rho_pi.sum(),1e-12)),
               rho_pi_max=float(rho_pi.max()), A_pi_max=float(A.max()),
               n_pi_atoms=len(pia))
    bars = pidt.persistence(A, shape)
    rec.update(pidt.descriptors(bars))
    bars_raw = pidt.persistence(rho_pi, shape)
    rec.update(pidt.descriptors(bars_raw, prefix='rawpi'))
    for th in (0.002, 0.005, 0.010):
        dd = pidt.descriptors(bars, prefix=f'th{int(th*1000):03d}', eps_noise=th)
        rec.update({k: v for k, v in dd.items() if k.endswith('_n')})
    rec['t_grid']=time.time()-t1
    rec['bars']={str(k):np.round(v,6).tolist() for k,v in bars.items()}

    # ---- baseline descriptors ----
    t2=time.time()
    rho_tot = pidt.eval_density(mol, dm_all, G)
    vmin,vmax,vmean,vvar,npts = pidt.esp_on_isosurface(mol, dm_all, G, rho_tot)
    rec.update(esp_min=vmin, esp_max=vmax, esp_mean=vmean, esp_var=vvar, esp_npts=npts,
               vdw_area=pidt.vdw_surface_area(syms,xyz),
               n_aromatic_rings=int(rdMolDescriptors.CalcNumAromaticRings(rd)),
               n_rings=int(rdMolDescriptors.CalcNumRings(rd)),
               tpsa=float(rdMolDescriptors.CalcTPSA(rd)),
               mw=float(Descriptors.MolWt(rd)),
               logp=float(Descriptors.MolLogP(rd)),
               t_baseline=time.time()-t2)
    rec['t_total']=time.time()-t0
    return rec

for name, smi, group in SET_A:
    if name in res:
        print('skip', name, flush=True); continue
    try:
        r = analyse(name, smi, group); res[name]=r
        json.dump(res, open(OUT,'w'), indent=1)
        print('%-24s nh=%2d nao=%3d %-8s Npi=%7.3f H0n=%3d H1n=%3d H2n=%2d Aint=%7.3f occl=%.2f t=%5.1fs'
              % (name, r['nheavy'], r['nao'], r['partition'],
                 r.get('Npi_sym', r.get('Npi_poav',float('nan'))), r['piIDT0_n'], r['piIDT1_n'],
                 r['piIDT2_n'], r['Api_integral'], r['occlusion_ratio'], r['t_total']), flush=True)
    except Exception as e:
        print('FAIL', name, type(e).__name__, str(e)[:150], flush=True)
        traceback.print_exc()

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import pidt
from rdkit import Chem

t0=time.time()
syms, xyz, rd = pidt.geometry_from_smiles('c1ccccc1')
xyz, rmsd = pidt.align_to_plane(syms, xyz)
mol, mf = pidt.run_dft(syms, xyz)
dm = mf.make_rdm1()
dmp, idx, leak = pidt.pi_density_matrix_symmetry(mol, dm)

pi_atoms = set(i for i,a in enumerate(rd.GetAtoms()) if a.GetIsAromatic() or str(a.GetHybridization()) in ('SP2','SP'))
print('pi atoms:', sorted(pi_atoms))

for h in (0.30, 0.20, 0.15):
    G, shape, ax = pidt.make_grid(xyz, pad=3.5, h=h)
    t=time.time()
    rho = pidt.eval_density(mol, dmp, G)
    S   = pidt.accessibility_mask(G, syms, xyz, pi_atoms)
    A   = rho*S
    bars = pidt.persistence(A, shape)
    d = pidt.descriptors(bars, voxel=(h/pidt.BOHR)**3, field=A)
    print('h=%.2f  npts=%d  t=%.1fs  rho_pi max=%.4f  Npi(grid)=%.3f' %
          (h, len(G), time.time()-t, rho.max(), rho.sum()*(h/pidt.BOHR)**3))
    print('   H0 n=%d totp=%.4f  | H1 n=%d totp=%.4f L1=%.4f | H2 n=%d totp=%.4f' %
          (d['piIDT0_n'],d['piIDT0_totpers'],d['piIDT1_n'],d['piIDT1_totpers'],d['piIDT1_L1'],
           d['piIDT2_n'],d['piIDT2_totpers']))
    print('   top H0 bars:', np.round(bars[0][np.argsort(-(bars[0][:,0]-bars[0][:,1]))][:4],4).tolist())
    print('   top H1 bars:', np.round(bars[1][np.argsort(-(bars[1][:,0]-bars[1][:,1]))][:4],4).tolist())
print('total %.1fs'%(time.time()-t0))

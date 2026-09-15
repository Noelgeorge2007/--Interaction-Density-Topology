import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import sys, time, numpy as np
import pidt

t0=time.time()
syms, xyz, rd = pidt.geometry_from_smiles('c1ccccc1')
xyz, rmsd = pidt.align_to_plane(syms, xyz)
print('planarity rmsd (A) = %.4f' % rmsd)
mol, mf = pidt.run_dft(syms, xyz, basis='6-31G*', xc='b3lyp')
print('E =', mf.e_tot, ' nao =', mol.nao, ' t=%.1fs'%(time.time()-t0))
dm = mf.make_rdm1()

par = pidt.ao_z_parity(mol)
print('AO parity counts: sigma=%d pi=%d mixed=%d' % ((par==1).sum(),(par==-1).sum(),(par==0).sum()))
dmp, idx, leak = pidt.pi_density_matrix_symmetry(mol, dm)
S = mol.intor('int1e_ovlp')
print('N_total = %.4f  N_pi(sym) = %.4f  sigma-pi leakage |D| = %.2e' %
      (np.einsum('ij,ji->',dm,S), np.einsum('ij,ji->',dmp,S), leak))

dmp2, npi2 = pidt.pi_density_matrix_poav(mol, dm, syms, xyz, rd)
print('N_pi(POAV) = %.4f' % npi2)
print('||D_pi(sym) - D_pi(POAV)||_F / ||D_pi(sym)||_F = %.4f' %
      (np.linalg.norm(dmp-dmp2)/np.linalg.norm(dmp)))

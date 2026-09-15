import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import pidt
syms, xyz, rd = pidt.geometry_from_smiles('c1ccccc1')
xyz, rmsd = pidt.align_to_plane(syms, xyz)
mol, mf = pidt.run_dft(syms, xyz)
dmp,_,_ = pidt.pi_density_matrix_symmetry(mol, mf.make_rdm1())
pi_atoms = set(i for i,a in enumerate(rd.GetAtoms()) if a.GetIsAromatic())
print(f"{'h':>5} {'npts':>9} {'Npi':>7} {'H0n':>4} {'H0tp':>8} {'H1n':>4} {'H1tp':>8} {'H1L1':>8} {'t/s':>6}")
for h in (0.40,0.30,0.25,0.20,0.15,0.12,0.10):
    G,shape,ax = pidt.make_grid(xyz, pad=3.5, h=h)
    t=time.time()
    A = pidt.eval_density(mol,dmp,G)*pidt.accessibility_mask(G,syms,xyz,pi_atoms)
    b = pidt.persistence(A,shape); d = pidt.descriptors(b, voxel=(h/pidt.BOHR)**3, field=A)
    print(f"{h:5.2f} {len(G):9d} {A.sum()*(h/pidt.BOHR)**3:7.3f} {d['piIDT0_n']:4d} {d['piIDT0_totpers']:8.4f} "
          f"{d['piIDT1_n']:4d} {d['piIDT1_totpers']:8.4f} {d['piIDT1_L1']:8.5f} {time.time()-t:6.1f}")

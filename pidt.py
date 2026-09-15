"""
pi-Interaction Density Topology (pi-IDT)
Reference implementation.

Pipeline:
  geometry -> DFT -> sigma/pi partition of the one-particle density matrix
           -> pi-density on a Cartesian grid
           -> steric accessibility mask
           -> cubical-complex persistent homology of the superlevel filtration
           -> descriptors
"""
import numpy as np
from pyscf import gto, dft, lib
from pyscf.dft import numint
import gudhi

BOHR = 0.52917721092

# Bondi van der Waals radii (Angstrom), J. Phys. Chem. 68, 441 (1964)
VDW = {'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
       'S': 1.80, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98, 'P': 1.80, 'Si': 2.10}


# ----------------------------------------------------------------------
# 1. geometry
# ----------------------------------------------------------------------
def geometry_from_smiles(smiles, seed=0xf00d, maxIters=2000):
    """ETKDG embedding + MMFF94 optimisation. Returns (symbols, coords[A], mol_rd)."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    if AllChem.EmbedMolecule(m, ps) != 0:
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(m, ps)
    AllChem.MMFFOptimizeMolecule(m, maxIters=maxIters)
    conf = m.GetConformer()
    syms = [a.GetSymbol() for a in m.GetAtoms()]
    xyz = np.array([list(conf.GetAtomPosition(i)) for i in range(m.GetNumAtoms())])
    return syms, xyz, m


def planarity_rmsd(syms, xyz, heavy_only=True):
    """RMS deviation (A) of atoms from their best-fit plane."""
    heavy = np.array([i for i, s in enumerate(syms) if (s != 'H' or not heavy_only)])
    if len(heavy) < 3:
        heavy = np.arange(len(syms))
    P = xyz[heavy]
    P = P - P.mean(0)
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    n = Vt[-1]
    return float(np.sqrt(np.mean((P @ n) ** 2))), n


def align_to_plane(syms, xyz):
    """Rotate so the heavy-atom best-fit plane is z = 0 (centroid at origin)."""
    rmsd, n = planarity_rmsd(syms, xyz)
    hv = [i for i, s in enumerate(syms) if s != 'H']
    if len(hv) < 3:
        hv = list(range(len(syms)))
    c = xyz[hv].mean(0)
    X = xyz - c
    # build a right-handed frame with n as z
    a = np.array([1.0, 0, 0])
    if abs(np.dot(a, n)) > 0.9:
        a = np.array([0, 1.0, 0])
    e1 = np.cross(a, n); e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    R = np.vstack([e1, e2, n])
    return X @ R.T, rmsd


# ----------------------------------------------------------------------
# 2. electronic structure
# ----------------------------------------------------------------------
def run_dft(syms, xyz, basis='6-31G*', xc='wb97x-d3bj', verbose=0, df=True):
    atom = [[s, tuple(r)] for s, r in zip(syms, xyz)]
    mol = gto.M(atom=atom, basis=basis, unit='Angstrom', verbose=verbose, charge=0, spin=0)
    mf = dft.RKS(mol)
    if df:
        mf = mf.density_fit()
    mf.xc = xc
    mf.grids.level = 3
    mf.conv_tol = 1e-9
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('SCF not converged')
    return mol, mf


# ----------------------------------------------------------------------
# 3. sigma / pi partition
# ----------------------------------------------------------------------
def valence_dm(mf, core_thresh=-2.0):
    """
    Density matrix of the valence occupied orbitals only. Core shells are
    dropped because for second-row and heavier elements the core np subshell
    is antisymmetric with respect to the molecular plane and would otherwise
    contaminate rho_pi with a compact, chemically inert lone-pair-like blob
    (e.g. the S 2p_z core of thiophene contributes two spurious pi electrons).
    """
    e = mf.mo_energy
    occ = mf.mo_occ > 0
    val = occ & (e > core_thresh)
    C = mf.mo_coeff[:, val]
    n = mf.mo_occ[val]
    ncore = int(occ.sum() - val.sum())
    return (C * n) @ C.T, ncore


def ao_z_parity(mol, npts=600, seed=1, tol=1e-4):
    """
    Classify each AO by its parity under reflection through the plane z = 0.
    Returns +1 (symmetric, sigma) / -1 (antisymmetric, pi) / 0 (mixed).
    Determined numerically, so it is independent of basis convention
    (Cartesian or spherical GTOs).
    """
    rng = np.random.default_rng(seed)
    R = mol.atom_coords() * BOHR
    # sample around every nucleus, off the mirror plane
    pts = []
    per = max(4, npts // max(1, len(R)))
    for c in R:
        q = c + rng.normal(scale=0.9, size=(per, 3))
        q[:, 2] = c[2] + np.sign(rng.normal(size=per)) * (0.3 + np.abs(rng.normal(scale=0.7, size=per)))
        pts.append(q)
    pts = np.vstack(pts)
    mir = pts.copy(); mir[:, 2] *= -1.0
    a1 = numint.eval_ao(mol, pts)
    a2 = numint.eval_ao(mol, mir)
    par = np.zeros(mol.nao, dtype=int)
    for k in range(mol.nao):
        u, v = a1[:, k], a2[:, k]
        m = np.abs(u) > 1e-3 * np.abs(u).max()
        if m.sum() < 5:
            par[k] = 1; continue
        u, v = u[m], v[m]
        r = float(u @ v / (u @ u))
        res = np.linalg.norm(v - r * u) / np.linalg.norm(u)
        if res > 1e-3:
            par[k] = 0
        elif r > 1 - tol:
            par[k] = 1
        elif r < -1 + tol:
            par[k] = -1
        else:
            par[k] = 0
    return par


def pi_density_matrix_symmetry(mol, dm):
    """
    Exact sigma/pi separation for a planar molecule aligned with z = 0.
    The AO overlap and the closed-shell density matrix are block diagonal in the
    reflection parity, so the pi block of D is an exact sub-density.
    """
    par = ao_z_parity(mol)
    if np.any(par == 0):
        raise ValueError('molecule is not aligned with a mirror plane at z=0')
    idx = np.where(par == -1)[0]
    dmp = np.zeros_like(dm)
    dmp[np.ix_(idx, idx)] = dm[np.ix_(idx, idx)]
    # leakage check: norm of the sigma-pi off-diagonal block of D
    jdx = np.where(par == 1)[0]
    leak = np.linalg.norm(dm[np.ix_(idx, jdx)])
    return dmp, idx, leak


def pi_atom_populations(mol, dm_pi):
    """Loewdin atomic populations of the pi sub-density (electrons per atom)."""
    S = mol.intor('int1e_ovlp')
    w, V = np.linalg.eigh(S)
    Sh = V @ np.diag(np.sqrt(w)) @ V.T
    diag = np.diag(Sh @ dm_pi @ Sh)
    pop = np.zeros(mol.natm)
    for k, (aid, _sym, _nm, _m) in enumerate(mol.ao_labels(fmt=False)):
        pop[aid] += diag[k]
    return pop


def pi_atom_set(mol, dm_pi, thresh=0.15):
    """
    Atoms that actually carry pi density, taken directly from the computed
    sub-density rather than from a valence-bond heuristic. Atoms below the
    threshold are treated as steric occluders in the accessibility mask.
    """
    pop = pi_atom_populations(mol, dm_pi)
    return set(int(i) for i in np.where(pop > thresh)[0]), pop


def local_pi_axes(syms, xyz, rd_mol):
    """
    POAV-style local pi axis for every sp2/sp (or aromatic) heavy atom.
    Returns {atom_index: list of unit vectors}.
    """
    from rdkit import Chem
    axes = {}
    conf_nb = {i: [n.GetIdx() for n in a.GetNeighbors()] for i, a in enumerate(rd_mol.GetAtoms())}
    for i, a in enumerate(rd_mol.GetAtoms()):
        if a.GetSymbol() == 'H':
            continue
        hyb = a.GetHybridization()
        conj = (a.GetIsAromatic() or hyb == Chem.HybridizationType.SP2
                or hyb == Chem.HybridizationType.SP)
        # a saturated heavy atom carrying lone pairs and bonded to a conjugated
        # centre (an aryl halide, ether or amine substituent) donates a p_pi
        # lone pair to the pi system and must be given a local axis too
        lp = (a.GetSymbol() in ('F', 'Cl', 'Br', 'I', 'O', 'N', 'S')
              and any(nb.GetIsAromatic()
                      or nb.GetHybridization() in (Chem.HybridizationType.SP2,
                                                   Chem.HybridizationType.SP)
                      for nb in a.GetNeighbors()))
        if not (conj or lp):
            continue
        nb = conf_nb[i]
        v = np.array([xyz[j] - xyz[i] for j in nb], dtype=float)
        v = np.array([u / np.linalg.norm(u) for u in v])
        if len(nb) >= 3:
            _, s, Vt = np.linalg.svd(v - v.mean(0), full_matrices=True)
            axes[i] = [Vt[-1]]
        elif len(nb) == 2:
            c = np.cross(v[0], v[1])
            if np.linalg.norm(c) < 0.15:            # near-linear (sp): two pi axes
                b = v[0]
                t = np.array([1.0, 0, 0])
                if abs(np.dot(t, b)) > 0.9:
                    t = np.array([0, 1.0, 0])
                e1 = np.cross(b, t); e1 /= np.linalg.norm(e1)
                e2 = np.cross(b, e1)
                axes[i] = [e1, e2]
            else:
                axes[i] = [c / np.linalg.norm(c)]
        elif len(nb) == 1:
            j = nb[0]
            # fall through: use the plane of the neighbour's other neighbours
            vj = np.array([xyz[k] - xyz[j] for k in conf_nb[j] if k != i], dtype=float)
            if len(vj) >= 2:
                c = np.cross(vj[0], vj[1])
                axes[i] = [c / np.linalg.norm(c)]
    return axes


def pi_density_matrix_poav(mol, dm, syms, xyz, rd_mol):
    """
    General (curvature-tolerant) pi partition by Loewdin projection onto the
    space spanned by local p_pi atomic orbitals.
    """
    S = mol.intor('int1e_ovlp')
    w, V = np.linalg.eigh(S)
    Sh = V @ np.diag(np.sqrt(w)) @ V.T
    Sinvh = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
    axes = local_pi_axes(syms, xyz, rd_mol)

    # locate p-shell AO triplets per atom
    labels = mol.ao_labels(fmt=False)   # (atom_id, symbol, ao_name, m)
    cols = []
    from collections import defaultdict
    pshells = defaultdict(dict)
    for k, (aid, sym, nm, m) in enumerate(labels):
        if nm.endswith('p') and m in ('x', 'y', 'z'):
            pshells[(aid, nm)][m] = k
    for (aid, shell), d in sorted(pshells.items()):
        if aid not in axes or set(d) != {'x', 'y', 'z'}:
            continue
        for n in axes[aid]:
            v = np.zeros(mol.nao)
            v[d['x']], v[d['y']], v[d['z']] = n
            cols.append(v)
    if not cols:
        raise ValueError('no pi atomic orbitals identified')
    C = np.array(cols).T                       # AO basis
    Cl = Sh @ C                                # Loewdin-orthogonal basis
    Q, r = np.linalg.qr(Cl)
    keep = np.abs(np.diag(r)) > 1e-8
    Q = Q[:, keep]
    P = Q @ Q.T
    Dl = Sh @ dm @ Sh
    Dpl = P @ Dl @ P
    dmp = Sinvh @ Dpl @ Sinvh
    npi = float(np.trace(Dpl))
    return dmp, npi


# ----------------------------------------------------------------------
# 4. grid, density, accessibility mask
# ----------------------------------------------------------------------
def make_grid(xyz, pad=3.5, h=0.20):
    """
    Cartesian grid symmetric about the origin on every axis. Because the
    molecule has been translated so that the heavy-atom centroid (and, for a
    planar molecule, its mirror plane) sits at the origin, the plane z = 0 is
    a grid plane; rho_pi vanishes there exactly by symmetry, so the two faces
    of a planar pi system are never numerically fused.
    """
    ext = np.abs(xyz).max(0) + pad
    ax = [h * np.arange(-int(np.ceil(e / h)), int(np.ceil(e / h)) + 1) for e in ext]
    shape = tuple(len(a) for a in ax)
    G = np.stack(np.meshgrid(*ax, indexing='ij'), axis=-1).reshape(-1, 3)
    return G, shape, ax


def eval_density(mol, dm, G, chunk=200000):
    """G in Angstrom; returns rho in atomic units (e / bohr^3)."""
    Gb = G / BOHR
    rho = np.empty(len(G))
    for i in range(0, len(G), chunk):
        ao = numint.eval_ao(mol, Gb[i:i + chunk])
        rho[i:i + chunk] = numint.eval_rho(mol, ao, dm)
    return rho


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def accessibility_mask(G, syms, xyz, pi_atoms, delta=1.0, scale=1.0):
    """
    S(r) in [0,1]: 0 where r lies inside the van der Waals sphere of a non-pi
    atom, 1 beyond that sphere + delta, smooth in between. Product over atoms.
    """
    S = np.ones(len(G))
    for i, s in enumerate(syms):
        if i in pi_atoms:
            continue
        R = VDW.get(s, 1.7) * scale
        d = np.linalg.norm(G - xyz[i], axis=1)
        S *= _smoothstep((d - R) / delta)
    return S


# ----------------------------------------------------------------------
# 5. persistent homology
# ----------------------------------------------------------------------
def persistence(field, shape):
    """
    Superlevel-set persistent homology of A_pi via the sublevel filtration of
    -A_pi on a cubical complex. Bars are returned in A_pi units with
    birth > death (a feature is born at a high threshold and dies at a lower one).
    Essential classes are capped at 0, the infimum of A_pi.
    """
    F = -np.asarray(field, dtype=float).reshape(shape)
    cc = gudhi.CubicalComplex(top_dimensional_cells=F)
    cc.compute_persistence(homology_coeff_field=2)
    out = {0: [], 1: [], 2: []}
    for dim in (0, 1, 2):
        for b, d in cc.persistence_intervals_in_dimension(dim):
            B = -b
            D = 0.0 if not np.isfinite(d) else -d
            if B - D > 0:
                out[dim].append((B, D))
    return {k: np.array(v).reshape(-1, 2) for k, v in out.items()}


def descriptors(bars, prefix='piIDT', eps_noise=5e-3, voxel=None, field=None):
    """Rotation-invariant summaries of the persistence diagram."""
    d = {}
    for dim in (0, 1, 2):
        B = bars[dim]
        life = (B[:, 0] - B[:, 1]) if len(B) else np.array([])
        life = life[life > eps_noise]
        life = np.sort(life)[::-1]
        p = f'{prefix}{dim}'
        d[f'{p}_n'] = int(len(life))
        d[f'{p}_totpers'] = float(life.sum())
        d[f'{p}_maxpers'] = float(life[0]) if len(life) else 0.0
        d[f'{p}_meanpers'] = float(life.mean()) if len(life) else 0.0
        if len(life):
            q = life / life.sum()
            d[f'{p}_entropy'] = float(-(q * np.log(q)).sum())
            d[f'{p}_normentropy'] = float(d[f'{p}_entropy'] / np.log(len(life))) if len(life) > 1 else 0.0
        else:
            d[f'{p}_entropy'] = 0.0
            d[f'{p}_normentropy'] = 0.0
        for j in range(3):                       # pi-barcode vector
            d[f'{p}_L{j+1}'] = float(life[j]) if len(life) > j else 0.0
        if len(B):
            d[f'{p}_maxbirth'] = float(B[:, 0].max())
    if field is not None and voxel is not None:
        d[f'{prefix}_integral'] = float(field.sum() * voxel)
    return d


# ----------------------------------------------------------------------
# 6. baseline descriptors
# ----------------------------------------------------------------------
def esp_on_isosurface(mol, dm, G, rho_tot, lo=8e-4, hi=1.2e-3, max_pts=4000, seed=0):
    """
    Electrostatic potential sampled on the rho = 0.001 a.u. isodensity surface.
    Returns (V_min, V_max, V_mean, V_var, n_points) in atomic units.
    """
    m = (rho_tot > lo) & (rho_tot < hi)
    pts = G[m]
    if len(pts) == 0:
        return (np.nan,) * 4 + (0,)
    if len(pts) > max_pts:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(len(pts), max_pts, replace=False)]
    pb = pts / BOHR
    V = np.zeros(len(pb))
    for i in range(0, len(pb), 500):
        chunk = pb[i:i + 500]
        Vmat = mol.intor('int1e_grids', grids=chunk)          # electronic, negative
        V[i:i + 500] = -np.einsum('pij,ji->p', Vmat, dm)
    Z = mol.atom_charges().astype(float)
    R = mol.atom_coords()
    for z, r in zip(Z, R):
        V += z / np.linalg.norm(pb - r, axis=1)
    return float(V.min()), float(V.max()), float(V.mean()), float(V.var()), int(len(V))


def vdw_surface_area(syms, xyz, n_sphere=1000, scale=1.0, seed=0):
    """Solvent-excluded-free van der Waals surface area (A^2) by Shrake-Rupley."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n_sphere) + 0.5
    phi = np.arccos(1 - 2 * idx / n_sphere)
    theta = np.pi * (1 + 5 ** 0.5) * idx
    sph = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], 1)
    R = np.array([VDW.get(s, 1.7) * scale for s in syms])
    area = 0.0
    for i in range(len(syms)):
        p = xyz[i] + R[i] * sph
        acc = np.ones(n_sphere, bool)
        for j in range(len(syms)):
            if j == i:
                continue
            acc &= np.linalg.norm(p - xyz[j], axis=1) > R[j]
        area += 4 * np.pi * R[i] ** 2 * acc.mean()
    return float(area)


def canonical_frame(syms, xyz, pi_atoms=None):
    """
    Deterministic molecular frame: axes are the principal axes of the
    pi-atom coordinate covariance, ordered by decreasing eigenvalue, with signs
    fixed by the sign of the third moment along each axis. Applying this to any
    rigidly rotated copy of a molecule returns the same coordinates, so every
    grid-derived descriptor is invariant to the input orientation by
    construction.
    """
    idx = sorted(pi_atoms) if pi_atoms else [i for i, s in enumerate(syms) if s != 'H']
    P = np.asarray(xyz, float)
    c = P[idx].mean(0)
    X = P - c
    C = (X[idx].T @ X[idx]) / len(idx)
    w, V = np.linalg.eigh(C)
    order = np.argsort(-w)
    V = V[:, order]
    Y = X @ V
    for k in range(3):
        m3 = (Y[idx, k] ** 3).sum()
        if abs(m3) < 1e-8:
            m3 = (np.abs(Y[idx, k]) * np.sign(Y[idx, k])).sum()
        if m3 < 0:
            V[:, k] *= -1
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1
    return X @ V

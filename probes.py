"""Standardised probe complexes for the interaction-energy benchmark."""
import numpy as np
from pyscf import gto, dft
import pidt

BOHR = pidt.BOHR

def probe_site(syms, xyz, rd):
    """
    Returns (centre, unit normal) defining where a probe is placed:
    the centroid of the largest SSSR ring (lowest atom index breaks ties) and
    that ring's best-fit plane normal; for acyclic pi systems, the midpoint of
    the first C=C bond and its local pi axis.
    """
    ri = rd.GetRingInfo()
    rings = [r for r in ri.AtomRings()]
    if rings:
        rings.sort(key=lambda r: (-len(r), min(r)))
        r = list(rings[0])
        P = xyz[r]
        c = P.mean(0)
        _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
        n = Vt[-1]
    else:
        ax = pidt.local_pi_axes(syms, xyz, rd)
        keys = sorted(ax)
        i, j = keys[0], keys[1]
        c = 0.5 * (xyz[i] + xyz[j])
        n = ax[i][0]
    n = n / np.linalg.norm(n)
    # orient away from the bulk of the molecule
    if np.dot(xyz.mean(0) - c, n) > 0:
        n = -n
    return c, n


def make_probe(kind, c, n, xyz_ref, d):
    """Atoms (symbol, position) of the probe placed at distance d along n."""
    if kind == 'Na+':
        return [('Na', c + d * n)], 1, 0
    if kind == 'Cl-':
        return [('Cl', c + d * n)], -1, 0
    if kind == 'benzene':
        # rigid benzene, C-C 1.391 A, C-H 1.088 A, ring plane parallel to the
        # substrate plane, displaced d along n and 1.6 A laterally
        e1 = np.cross(n, [0, 0, 1.0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(n, [1.0, 0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        at = []
        for k in range(6):
            a = k * np.pi / 3
            u = np.cos(a) * e1 + np.sin(a) * e2
            at.append(('C', c + d * n + 1.6 * e1 + 1.391 * u))
            at.append(('H', c + d * n + 1.6 * e1 + 2.479 * u))
        return at, 0, 0
    if kind == 'CH4':
        e1 = np.cross(n, [0, 0, 1.0])
        if np.linalg.norm(e1) < 1e-6:
            e1 = np.cross(n, [1.0, 0, 0])
        e1 /= np.linalg.norm(e1); e2 = np.cross(n, e1)
        C = c + d * n
        at = [('C', C), ('H', C - 1.087 * n)]
        for k in range(3):
            a = k * 2 * np.pi / 3
            u = np.cos(a) * e1 + np.sin(a) * e2
            at.append(('H', C + 1.087 * (np.cos(np.deg2rad(70.53)) * n
                                         + np.sin(np.deg2rad(70.53)) * u)))
        return at, 0, 0
    raise ValueError(kind)


def _scf(atoms, ghost_flags, charge, basis, xc):
    spec = []
    for (s, r), gh in zip(atoms, ghost_flags):
        spec.append([('ghost-' + s) if gh else s, tuple(r)])
    mol = gto.M(atom=spec, basis=basis, unit='Angstrom', charge=charge,
                spin=0, verbose=0)
    mf = dft.RKS(mol).density_fit()
    mf.xc = xc; mf.conv_tol = 1e-9; mf.grids.level = 3
    e = mf.kernel()
    if not mf.converged:
        mf = mf.newton(); e = mf.kernel()
    return float(e), bool(mf.converged)


def interaction_energy(syms, xyz, probe_atoms, q_probe, basis='6-31+G*',
                       xc='wb97x-d3bj'):
    """Counterpoise-corrected interaction energy in kcal/mol."""
    A = [(s, r) for s, r in zip(syms, xyz)]
    B = probe_atoms
    nA, nB = len(A), len(B)
    full = A + B
    e_ab, c1 = _scf(full, [False] * (nA + nB), q_probe, basis, xc)
    e_a, c2 = _scf(full, [False] * nA + [True] * nB, 0, basis, xc)
    e_b, c3 = _scf(full, [True] * nA + [False] * nB, q_probe, basis, xc)
    return dict(e_int=(e_ab - e_a - e_b) * 627.5094740631,
                e_ab=e_ab, e_a=e_a, e_b=e_b, converged=bool(c1 and c2 and c3))

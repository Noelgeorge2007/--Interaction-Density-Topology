"""GFN2-xTB geometry optimisation via tblite + ASE."""
import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from ase.calculators.calculator import Calculator, all_changes
from tblite.interface import Calculator as TBCalc

HARTREE_EV = 27.211386245988
BOHR = 0.52917721092

class GFN2(Calculator):
    implemented_properties = ['energy', 'forces']
    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        num = np.array(atoms.get_atomic_numbers())
        pos = np.array(atoms.get_positions()) / BOHR
        calc = TBCalc('GFN2-xTB', num, pos)
        calc.set('verbosity', 0)
        r = calc.singlepoint()
        self.results['energy'] = float(r.get('energy')) * HARTREE_EV
        self.results['forces'] = -np.array(r.get('gradient')) * HARTREE_EV / BOHR

def optimise(syms, xyz, fmax=0.01, steps=500):
    at = Atoms(symbols=syms, positions=xyz)
    at.calc = GFN2()
    opt = BFGS(at, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    return np.array(at.get_positions()), bool(opt.converged())


def prepare_geometry(syms, xyz, n_restarts=3, sigma=0.12, seed=0, snap_tol=0.05):
    """
    GFN2-xTB relaxation with random out-of-plane seeding to avoid planar saddle
    points, followed by alignment of the heavy-atom best-fit plane with z = 0.
    If every atom then lies within `snap_tol` of that plane the coordinates are
    snapped onto it, which makes the sigma/pi reflection symmetry exact.
    """
    import pidt
    rng = np.random.default_rng(seed)
    best = None
    import pidt as _pp
    flat, _ = _pp.align_to_plane(syms, xyz); flat = flat.copy(); flat[:, 2] = 0.0
    starts = [xyz, flat] + [xyz + rng.normal(scale=sigma, size=xyz.shape)
                            for _ in range(max(0, n_restarts - 2))]
    for k, x0 in enumerate(starts):
        try:
            x1, conv = optimise(syms, x0)
        except Exception:
            continue
        at = Atoms(symbols=syms, positions=x1); at.calc = GFN2()
        e = at.get_potential_energy()
        cand = (x1, e, conv)
        if best is None or e < best[1] - 0.02:
            best = cand
        elif e < best[1] + 0.02:
            # near-degenerate (< 0.02 eV): prefer the flatter structure, since
            # the out-of-plane potential of large pi systems is very shallow
            import pidt as _p
            r_new = _p.planarity_rmsd(syms, x1, heavy_only=False)[0]
            r_old = _p.planarity_rmsd(syms, best[0], heavy_only=False)[0]
            if r_new < r_old - 1e-3:
                best = cand
    if best is None:
        raise RuntimeError('xTB optimisation failed')
    x, e, conv = best
    x, rmsd_heavy = pidt.align_to_plane(syms, x)
    rmsd_all, _ = pidt.planarity_rmsd(syms, x, heavy_only=False)
    snapped = False
    if np.abs(x[:, 2]).max() < snap_tol:
        x = x.copy(); x[:, 2] = 0.0
        snapped = True
    return dict(xyz=x, e_xtb=float(e), converged=conv,
                planarity_heavy=float(rmsd_heavy), planarity_all=float(rmsd_all),
                max_out_of_plane=float(np.abs(x[:, 2]).max()), snapped=snapped)

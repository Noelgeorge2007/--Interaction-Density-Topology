import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths
import sys, os, json, time, traceback
import numpy as np, pidt, probes
from molecules import SET_A

OUT   = str(paths.RESULTS / 'setB.json')
SETA  = str(paths.RESULTS / 'setA.json')
BASIS = os.environ.get('PIDT_BASIS_INT','6-31+G*')
XC    = os.environ.get('PIDT_XC','wb97x-d3bj')
MAXH  = int(os.environ.get('PIDT_MAXHEAVY','12'))

PROBES = [('Na+', 2.6), ('Cl-', 3.2), ('benzene', 3.4), ('CH4', 3.6)]
SCAN   = {'Na+': [2.2, 2.6, 3.0], 'Cl-': [2.8, 3.2, 3.6]}
SCAN_SUBSET = ['benzene','pyridine','s-triazine','pyrrole','thiophene',
               'hexafluorobenzene','nitrobenzene','naphthalene']

setA = json.load(open(SETA))
res  = json.load(open(OUT)) if os.path.exists(OUT) else {}

for name, smi, group in SET_A:
    if name not in setA:            continue
    r = setA[name]
    if r['nheavy'] > MAXH:          continue
    syms = r['symbols']; xyz = np.array(r['coords'])
    from rdkit import Chem
    rd = Chem.AddHs(Chem.MolFromSmiles(smi))
    rec = res.get(name, dict(name=name, smiles=smi, group=group))
    try:
        c, n = probes.probe_site(syms, xyz, rd)
        rec['probe_centre'] = c.tolist(); rec['probe_normal'] = n.tolist()
        for kind, d in PROBES:
            key = f'eint_{kind}'
            if key in rec: continue
            t = time.time()
            pa, q, _ = probes.make_probe(kind, c, n, xyz, d)
            out = probes.interaction_energy(syms, xyz, pa, q, basis=BASIS, xc=XC)
            rec[key] = out['e_int']; rec[key+'_conv'] = out['converged']
            rec[key+'_d'] = d
            print('%-22s %-8s d=%.2f  Eint=%8.3f kcal/mol  conv=%s  %5.1fs'
                  % (name, kind, d, out['e_int'], out['converged'], time.time()-t), flush=True)
            res[name] = rec; json.dump(res, open(OUT,'w'), indent=1)
        if name in SCAN_SUBSET:
            for kind, ds in SCAN.items():
                k = f'scan_{kind}'
                if k in rec: continue
                curve = []
                for d in ds:
                    pa, q, _ = probes.make_probe(kind, c, n, xyz, d)
                    o = probes.interaction_energy(syms, xyz, pa, q, basis=BASIS, xc=XC)
                    curve.append([d, o['e_int']])
                rec[k] = curve
                x = np.array([p[0] for p in curve]); y = np.array([p[1] for p in curve])
                co = np.polyfit(x, y, 2)
                dmin = -co[1]/(2*co[0]) if co[0] > 0 else float('nan')
                rec[k+'_dmin'] = float(dmin)
                rec[k+'_emin'] = float(np.polyval(co, dmin)) if np.isfinite(dmin) else float('nan')
                print('   scan %-4s %s -> dmin=%.2f Emin=%.3f' %
                      (kind, np.round(y,2).tolist(), dmin, rec[k+'_emin']), flush=True)
                res[name] = rec; json.dump(res, open(OUT,'w'), indent=1)
    except Exception as e:
        print('FAIL', name, type(e).__name__, str(e)[:150], flush=True)
        traceback.print_exc()
    res[name] = rec; json.dump(res, open(OUT,'w'), indent=1)
print('done')

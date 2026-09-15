#!/usr/bin/env python3
"""Check reference-data completeness and independently recompute headline summaries."""
import ast
import hashlib
import json
import math
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def read(name):
    return json.loads((ROOT / 'results' / name).read_text())
def require(ok, message):
    if not ok:
        raise ValueError(message)
A, B = read('setA.json'), read('setB.json')
# Read the literal molecule definitions without importing computational code.
tree = ast.parse((ROOT / 'src/molecules.py').read_text())
sets = {}
for node in tree.body:
    if isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == 'SET_A':
                sets[target.id] = ast.literal_eval(node.value)
require(set(A) == {row[0] for row in sets['SET_A']}, 'Set A records differ from declared molecules')
require(len(B) == 26 and set(B) <= set(A), 'Set B membership/count mismatch')
for name, rec in A.items():
    require(len(rec['coords']) == len(rec['symbols']), f'{name}: coordinates/symbols mismatch')
    require(all(len(x)==3 and all(math.isfinite(v) for v in x) for x in rec['coords']), f'{name}: invalid coordinates')
    for dim in ('0','1','2'):
        require(all(len(x)==2 and all(math.isfinite(v) for v in x) and x[0]>=x[1] for x in rec['bars'][dim]), f'{name}: invalid persistence bars')
    require(0 <= rec['occlusion_ratio'] <= 1, f'{name}: invalid occlusion')
for name, rec in B.items():
    for target in ('eint_Na+', 'eint_Cl-', 'eint_benzene', 'eint_CH4'):
        require(target in rec and math.isfinite(rec[target]), f'{name}: missing {target}')
NR = read('nring.json')
match = sum(r['piIDT1_n'] == 2*NR[n] for n,r in A.items())
N = read('numbers.json')
require(match == int(N['nRingMatchB']), 'Ring summary mismatch')
SP = read('shield_poav.json')
drop=100*(1-SP['hexaethylbenzene']['I']/SP['benzene']['I'])
require(f'{drop:.0f}' == N['shieldDrop'], 'Shielding summary mismatch')
G = read('regression.json')
for t,g in G.items():
    require(set(g['names'])==set(B) and len(g['y'])==26, f'{t}: regression sample mismatch')
    require(all(abs(y-B[n][t])<1e-10 for n,y in zip(g['names'],g['y'])), f'{t}: target mismatch')
    require(g['baseline']['q2_mean'] > g['baseline+piIDT']['q2_mean'], f'{t}: headline negative comparison differs')
for p in ROOT.rglob('*.json'):
    json.loads(p.read_text())
manifest=ROOT/'MANIFEST.sha256'
if manifest.exists():
    for line in manifest.read_text().splitlines():
        digest,name=line.split('  ',1)
        p=ROOT/name
        require(p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==digest, f'Checksum mismatch: {name}')
print(f'PASS: {len(A)} descriptor records; {len(B)} molecules x 4 targets; {match}/{len(A)} ring matches; shielding reduction {drop:.6f}%; reference manifest verified when present.')

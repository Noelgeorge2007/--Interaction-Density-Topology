"""Repository paths, resolved from this file's location.

Every script in src/ imports ROOT from here, so the tree can be unpacked
anywhere. Individual directories can still be overridden by environment
variable, which is how the long DFT runs were split across sessions:

    PIDT_ROOT      whole tree           (default: the parent of src/)
    PIDT_RESULTS   JSON and CSV output  (default: $PIDT_ROOT/results)
    PIDT_FIGURES   rendered figures     (default: $PIDT_ROOT/figures)
    PIDT_PAPER     LaTeX sources        (default: $PIDT_ROOT/paper)
"""
import os
from pathlib import Path

SRC     = Path(__file__).resolve().parent
ROOT    = Path(os.environ.get('PIDT_ROOT', SRC.parent))
RESULTS = Path(os.environ.get('PIDT_RESULTS', ROOT / 'results'))
FIGURES = Path(os.environ.get('PIDT_FIGURES', ROOT / 'figures'))
PAPER   = Path(os.environ.get('PIDT_PAPER',   ROOT / 'paper'))
SUBMISSION = ROOT / 'submission'

for _d in (RESULTS, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

"""Repository-relative paths for code, calculation results and analysis outputs."""
import os
from pathlib import Path
SRC = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('PIDT_ROOT', SRC.parent))
RESULTS = Path(os.environ.get('PIDT_RESULTS', ROOT / 'results'))
FIGURES = Path(os.environ.get('PIDT_FIGURES', ROOT / 'figures'))
ANALYSIS = Path(os.environ.get('PIDT_ANALYSIS', ROOT / 'analysis_outputs'))
for directory in (RESULTS, FIGURES, ANALYSIS / 'tables', ANALYSIS / 'figs'):
    directory.mkdir(parents=True, exist_ok=True)

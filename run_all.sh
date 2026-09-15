#!/usr/bin/env bash
# Cached presentation, regression refitting, or electronic-structure workflow.
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$R"
export PYTHONPATH="$R/src" OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
mode="${1:-cached}"
case "$mode" in cached|analysis|all) ;; *) echo 'Usage: bash run_all.sh [cached|analysis|all]' >&2; exit 2;; esac
mkdir -p "${PIDT_RESULTS:-$R/results}" "${PIDT_ANALYSIS:-$R/analysis_outputs}/tables" "${PIDT_ANALYSIS:-$R/analysis_outputs}/figs"
if [ "$mode" = all ]; then
  for stage in run_setA run_setB run_robust run_esp run_extra run_extra2; do
    python3 "src/$stage.py" 2>&1 | tee "${PIDT_RESULTS:-$R/results}/$stage.log"
  done
fi
if [ "$mode" != cached ]; then
  python3 src/regress.py
  python3 src/regress2.py
fi
# make_aux needs setB_table.csv, supplied for cached mode and generated above otherwise.
python3 src/make_aux.py
python3 src/analysis.py
python3 src/analysis2.py
python3 src/export_figs.py
printf '\nTables and numerical macros: analysis_outputs/. Figures: figures/.\n'

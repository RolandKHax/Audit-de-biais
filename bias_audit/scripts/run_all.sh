#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [ -x "venv/bin/python" ]; then
  PYTHON_BIN="venv/bin/python"
fi

"${PYTHON_BIN}" -u scripts/download_compas.py

"${PYTHON_BIN}" -u -m src.audit_main \
  --data data/raw/compas-scores-two-years.csv \
  --preset compas \
  --protected-attrs race sex \
  --label two_year_recid \
  --debiasing reweighting resampling threshold fairlearn_demographic_parity fairlearn_equalized_odds adversarial_pytorch \
  --models logistic_regression random_forest mlp \
  --feature-policies without_sensitive with_sensitive \
  --seeds 0 1 2 3 4 \
  --bootstrap-iterations 200 \
  --adversarial-epochs 10 \
  --output reports/audit_report.html \
  --output-dir reports/figures \
  --results-dir results

"${PYTHON_BIN}" -u scripts/generate_latex_deliverables.py

if command -v pdflatex >/dev/null 2>&1; then
  (
    cd reports/latex
    pdflatex -interaction=nonstopmode rapport_audit_biais.tex
    bibtex rapport_audit_biais || true
    pdflatex -interaction=nonstopmode rapport_audit_biais.tex
    pdflatex -interaction=nonstopmode rapport_audit_biais.tex
  )
  (
    cd slides
    pdflatex -interaction=nonstopmode presentation.tex
    pdflatex -interaction=nonstopmode presentation.tex
  )
else
  echo "pdflatex indisponible: les fichiers .tex ont été générés sans PDF."
fi

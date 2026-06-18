"""Exécute l'audit fairness COMPAS avec mitigations non adversariales."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.audit_main import BiasAuditor


def main():
    auditor = BiasAuditor({
        "data_path": "data/raw/compas-scores-two-years.csv",
        "preset": "compas",
        "protected_attrs": ["race", "sex"],
        "label_name": "two_year_recid",
        "debiasing_methods": [
            "reweighting",
            "resampling",
            "threshold",
            "fairlearn_demographic_parity",
            "fairlearn_equalized_odds",
        ],
        "models": ["logistic_regression", "random_forest", "mlp"],
        "feature_policies": ["without_sensitive", "with_sensitive"],
        "seeds": [0, 1, 2, 3, 4],
        "output_path": "reports/audit_report.html",
        "output_dir": "reports/figures",
        "results_dir": "results",
    })
    auditor.run()


if __name__ == "__main__":
    main()

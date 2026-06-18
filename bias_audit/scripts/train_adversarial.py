"""Exécute l'audit adversarial PyTorch sur COMPAS, avec skip propre si torch manque."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.advanced_mitigation import torch_available
from src.audit_main import BiasAuditor


def main():
    if not torch_available():
        print("SKIP: PyTorch n'est pas installé dans cet environnement.")
        return

    auditor = BiasAuditor({
        "data_path": "data/raw/compas-scores-two-years.csv",
        "preset": "compas",
        "protected_attrs": ["race", "sex"],
        "label_name": "two_year_recid",
        "debiasing_methods": ["adversarial_pytorch"],
        "models": ["logistic_regression"],
        "feature_policies": ["without_sensitive"],
        "seeds": [0],
        "output_path": "reports/audit_report.html",
        "output_dir": "reports/figures",
        "results_dir": "results",
        "adversarial_epochs": 20,
    })
    auditor.run()


if __name__ == "__main__":
    main()

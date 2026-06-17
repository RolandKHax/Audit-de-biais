# Fichier: tests/test_audit_main.py
"""
Smoke test du workflow CLI sans passer par un sous-processus.
"""

import matplotlib

matplotlib.use("Agg")

from src.audit_main import BiasAuditor
from src.data_processing import generate_sample_data


def test_bias_auditor_end_to_end(tmp_path):
    data_path = tmp_path / "dataset.csv"
    report_path = tmp_path / "audit_report.html"
    figures_dir = tmp_path / "figures"

    generate_sample_data(n_samples=500, random_state=42).to_csv(data_path, index=False)

    auditor = BiasAuditor({
        "data_path": str(data_path),
        "model_path": None,
        "protected_attrs": ["gender", "race"],
        "label_name": "label",
        "debiasing_methods": ["reweighting", "resampling"],
        "output_path": str(report_path),
        "output_dir": str(figures_dir),
    })

    auditor.run()

    assert report_path.exists()
    assert report_path.with_suffix(".json").exists()
    assert (figures_dir / "debiasing_comparison.png").exists()
    assert "gender" in auditor.results["baseline_evaluation"]["fairness"]
    assert "reweighting" in auditor.results["debiasing_results"]["gender"]
    assert "resampling" in auditor.results["debiasing_results"]["gender"]


def test_bias_auditor_resampling_only(tmp_path):
    data_path = tmp_path / "dataset.csv"
    report_path = tmp_path / "audit_report.html"
    figures_dir = tmp_path / "figures"

    generate_sample_data(n_samples=500, random_state=42).to_csv(data_path, index=False)

    auditor = BiasAuditor({
        "data_path": str(data_path),
        "model_path": None,
        "protected_attrs": ["gender", "race"],
        "label_name": "label",
        "debiasing_methods": ["resampling"],
        "output_path": str(report_path),
        "output_dir": str(figures_dir),
    })

    auditor.run()

    assert report_path.exists()
    assert "resampling" in auditor.results["debiasing_results"]["gender"]
    assert "reweighting" not in auditor.results["debiasing_results"]["gender"]

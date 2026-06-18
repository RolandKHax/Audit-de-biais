# Guide de démarrage rapide

## 1. Installer

```bash
cd "Audit de biais d’un modèle de classification/bias_audit"
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## 2. Lancer tout le projet

```bash
bash scripts/run_all.sh
```

## 3. Vérifier les sorties

Rapports :

- `reports/latex/rapport_audit_biais.pdf`
- `slides/presentation.pdf`
- `reports/audit_report.html`
- `reports/audit_report.json`

Résultats :

- `results/tables/baseline_summary_aggregate.csv`
- `results/tables/fairness_by_group.csv`
- `results/tables/mitigation_summary.csv`
- `results/metrics/audit_metrics.json`

## 4. Commandes séparées

```bash
python scripts/download_compas.py
python scripts/train_baseline.py
python scripts/evaluate_fairness.py
python scripts/train_adversarial.py
python scripts/generate_latex_deliverables.py
```

## 5. Tests

```bash
python -m compileall -q src scripts tests
python -m pytest tests -q
```

## Note importante

Le dataset synthétique reste disponible pour les tests unitaires, mais le livrable principal utilise COMPAS réel comme demandé dans le cahier de charges.

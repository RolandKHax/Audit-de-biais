# Instructions d'execution du projet

Ce depot contient un projet executable d'audit de biais pour classification binaire. Le workflow principal est maintenant aligne sur Python 3.14 et sur les dependances listees dans `requirements.txt`.

## Installation

```bash
cd bias_audit
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Verification:

```bash
python -m pip check
python -c "from src.data_processing import DataProcessor; from src.metrics import FairnessMetrics; print('imports OK')"
```

## Execution CLI

```bash
python -c "from src.data_processing import generate_sample_data; generate_sample_data(5000).to_csv('data/processed/dataset.csv', index=False)"

python -m src.audit_main \
  --data data/processed/dataset.csv \
  --protected-attrs gender race \
  --label label \
  --debiasing reweighting resampling \
  --output reports/audit_report.html \
  --output-dir reports/figures
```

Le CLI entraine un modele baseline si `--model` n'est pas fourni.

## Execution notebooks

```bash
jupyter nbconvert --execute notebooks/*.ipynb --to notebook --inplace
```

Ordre logique:

1. `01_EDA.ipynb`
2. `02_baseline_evaluation.ipynb`
3. `03_debiasing_experiments.ipynb`
4. `04_final_analysis.ipynb`

## Tests

```bash
python -m pytest tests -q
python -m compileall -q src tests
```

## Livrables generes

- Rapport CLI: `reports/audit_report.html`
- Details CLI: `reports/audit_report.json`
- Figures: `reports/figures/`
- Synthese executive: `reports/executive_summary.txt`
- Model Card: `reports/MODEL_CARD.md`
- Recommandations: `reports/strategic_recommendations.txt`
- Resultats experimentaux: `outputs/debiasing_results.json`

## Remarques de coherence

- `configs/config.yaml` est une configuration de reference, pas un fichier charge automatiquement par le CLI.
- Les methodes CLI supportees sont `reweighting` et `resampling`.
- Le module contient une classe adversariale optionnelle, mais TensorFlow n'est pas installe par defaut afin de conserver la compatibilite Python 3.14.
- La compilation LaTeX est possible si une distribution LaTeX locale est installee.

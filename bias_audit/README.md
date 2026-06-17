# Audit de biais - modele de classification

Ce projet audite un modele de classification binaire, mesure les ecarts de fairness par groupe protege, applique des mitigations compatibles avec Python 3.14, puis genere des rapports et figures reproductibles.

## Ce que le projet couvre

- Exploration des donnees et detection de proxies.
- Evaluation de performance: accuracy, precision, recall, F1, ROC-AUC.
- Evaluation de fairness: demographic parity, disparate impact, equal opportunity, average odds, predictive parity.
- Mitigation compatible Python 3.14: reweighting et resampling.
- Sorties: rapport HTML, rapport JSON, figures, notebooks executes, synthese executive et Model Card.

Les metriques de fairness sont implementees dans `src/metrics.py`. Les dependances lourdes ou non compatibles Python 3.14 par defaut, comme AIF360, Fairlearn et TensorFlow, ne sont pas requises pour executer le projet.

## Installation

```bash
cd bias_audit
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip check
```

Important: la commande correcte est `pip install -r requirements.txt`. La forme `pip install - requirements.txt` est invalide.

## Structure

```text
bias_audit/
├── configs/          # configuration de reference
├── data/             # donnees brutes et traitees
├── models/           # modeles sauvegardes
├── notebooks/        # workflow notebook en 4 etapes
├── reports/          # rapports, figures, LaTeX
├── src/              # code source
├── tests/            # tests unitaires et smoke tests
└── outputs/          # sorties experimentales intermediaires
```

## Execution rapide

Generer un jeu de donnees synthetique:

```bash
python -c "from src.data_processing import generate_sample_data; generate_sample_data(5000).to_csv('data/processed/dataset.csv', index=False)"
```

Lancer l'audit complet:

```bash
python -m src.audit_main \
  --data data/processed/dataset.csv \
  --protected-attrs gender race \
  --label label \
  --debiasing reweighting resampling \
  --output reports/audit_report.html \
  --output-dir reports/figures
```

La commande `python src/audit_main.py ...` fonctionne aussi, mais `python -m src.audit_main` est la forme recommandee depuis la racine `bias_audit`.

## Notebooks

Les notebooks sont executables dans cet ordre:

1. `notebooks/01_EDA.ipynb`
2. `notebooks/02_baseline_evaluation.ipynb`
3. `notebooks/03_debiasing_experiments.ipynb`
4. `notebooks/04_final_analysis.ipynb`

Execution en ligne de commande:

```bash
jupyter nbconvert --execute notebooks/*.ipynb --to notebook --inplace
```

## Tests

```bash
python -m pytest tests -q
python -m compileall -q src tests
```

## Rapport LaTeX

Le rapport source se trouve dans `reports/latex/rapport_audit_biais.tex`.

```bash
cd reports/latex
pdflatex rapport_audit_biais.tex
bibtex rapport_audit_biais
pdflatex rapport_audit_biais.tex
pdflatex rapport_audit_biais.tex
```

La compilation LaTeX necessite une distribution LaTeX locale.

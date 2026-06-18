# Audit de biais d'un modèle de classification - COMPAS

Projet complet d'audit de biais sur le dataset réel COMPAS ProPublica. Le workflow mesure les disparités liées à `race` et `sex`, compare plusieurs modèles, applique des méthodes d'atténuation et génère un rapport final ainsi qu'une présentation Beamer.

## Couverture du cahier de charges

- Dataset principal réel : `data/raw/compas-scores-two-years.csv`.
- Dataset traité : `data/processed/compas_processed.csv`.
- Baselines : Logistic Regression, Random Forest, MLP.
- Comparaison avec et sans attributs sensibles.
- Métriques : accuracy, precision, recall, F1, ROC-AUC, demographic parity, disparate impact, equalized odds, equal opportunity, predictive parity, TPR/FPR, selection rate.
- Mitigation : reweighting, resampling, Fairlearn reductions, adversarial debiasing PyTorch, post-processing par seuils.
- Protocole : train/validation/test stratifié et seeds `0 1 2 3 4`.
- Livrables : scripts, notebooks, tableaux CSV, JSON, figures, rapport PDF, slides PDF, recommandations.

## Installation

```bash
cd "Audit de biais d’un modèle de classification/bias_audit"
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Note : AIF360 est documenté comme toolkit de référence, mais n'est pas requis par défaut afin d'éviter de bloquer l'exécution sur des dépendances binaires incompatibles.

## Exécution complète

```bash
bash scripts/run_all.sh
```

Cette commande :

1. télécharge COMPAS ;
2. génère `data/processed/compas_processed.csv` ;
3. lance l'audit multi-seeds ;
4. applique les mitigations ;
5. génère les tableaux dans `results/tables/` ;
6. génère les métriques dans `results/metrics/` ;
7. génère `reports/latex/rapport_audit_biais.tex` et `slides/presentation.tex` ;
8. compile les PDF si `pdflatex` est disponible.

## Scripts principaux

```bash
python scripts/download_compas.py
python scripts/train_baseline.py
python scripts/evaluate_fairness.py
python scripts/train_adversarial.py
```

Commande CLI équivalente au run complet :

```bash
python -m src.audit_main \
  --data data/raw/compas-scores-two-years.csv \
  --preset compas \
  --protected-attrs race sex \
  --label two_year_recid \
  --debiasing reweighting resampling threshold fairlearn_demographic_parity fairlearn_equalized_odds adversarial_pytorch \
  --models logistic_regression random_forest mlp \
  --feature-policies without_sensitive with_sensitive \
  --seeds 0 1 2 3 4 \
  --output reports/audit_report.html \
  --output-dir reports/figures \
  --results-dir results
```

## Notebooks

Les notebooks suivent la feuille de route attendue :

1. `notebooks/01_EDA_bias_analysis.ipynb`
2. `notebooks/02_baseline_model.ipynb`
3. `notebooks/03_fairness_metrics.ipynb`
4. `notebooks/04_mitigation_resampling.ipynb`
5. `notebooks/05_adversarial_debiasing_pytorch.ipynb`
6. `notebooks/06_comparison_recommendations.ipynb`

## Livrables générés

- `reports/audit_report.html`
- `reports/audit_report.json`
- `reports/latex/rapport_audit_biais.pdf`
- `slides/presentation.pdf`
- `results/tables/baseline_summary.csv`
- `results/tables/baseline_summary_aggregate.csv`
- `results/tables/fairness_by_group.csv`
- `results/tables/mitigation_summary.csv`
- `results/metrics/audit_metrics.json`
- `reports/figures/*.png`

## Validation

```bash
python -m compileall -q src scripts tests
python -m pytest tests -q
```

Compilation manuelle des PDF :

```bash
cd reports/latex
pdflatex rapport_audit_biais.tex
bibtex rapport_audit_biais || true
pdflatex rapport_audit_biais.tex
pdflatex rapport_audit_biais.tex

cd ../../slides
pdflatex presentation.tex
pdflatex presentation.tex
```

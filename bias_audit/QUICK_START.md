# Guide de demarrage rapide

## 1. Installer

```bash
cd bias_audit
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip check
```

Le projet cible Python 3.14 avec des dependances disponibles dans cet environnement. AIF360, Fairlearn, TensorFlow et les anciens builds PyTorch ne sont pas necessaires pour le workflow par defaut.

## 2. Generer des donnees de test

```bash
python -c "from src.data_processing import generate_sample_data; generate_sample_data(5000).to_csv('data/processed/dataset.csv', index=False)"
```

## 3. Lancer l'audit CLI

```bash
python -m src.audit_main \
  --data data/processed/dataset.csv \
  --protected-attrs gender race \
  --label label \
  --debiasing reweighting resampling \
  --output reports/audit_report.html \
  --output-dir reports/figures
```

Sorties principales:

- `reports/audit_report.html`
- `reports/audit_report.json`
- `reports/figures/*.png`

## 4. Lancer les notebooks

```bash
jupyter nbconvert --execute notebooks/*.ipynb --to notebook --inplace
```

Les notebooks generent aussi:

- `reports/executive_summary.txt`
- `reports/MODEL_CARD.md`
- `reports/strategic_recommendations.txt`
- `reports/final_audit_report.json`
- `outputs/debiasing_results.json`

## 5. Tester

```bash
python -m pytest tests -q
python -m compileall -q src tests
```

## Donnees attendues

Le CSV doit contenir:

- une colonne cible binaire, par exemple `label`;
- une ou plusieurs colonnes protegees, par exemple `gender`, `race`;
- des variables predictives numeriques ou categorielles.

Exemple:

```csv
gender,race,age,education,experience,score,label
Male,White,35,Bachelor,10,85,1
Female,Black,28,Master,5,78,0
```

## Configuration

`configs/config.yaml` sert de reference projet. Le CLI lit explicitement les arguments passes en ligne de commande; gardez donc les commandes ci-dessus comme source d'execution principale.

Les mitigations activees par defaut sont:

- `reweighting`
- `resampling`

Les approches adversariales et certains post-traitements restent des extensions possibles, mais ne sont pas activees dans l'environnement Python 3.14 par defaut.

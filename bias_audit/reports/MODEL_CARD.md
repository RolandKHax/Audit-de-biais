# Model Card - Audit de biais

## Usage prevu
Classification binaire sur donnees tabulaires synthetiques ou metier au meme format.

## Donnees
Attributs proteges audites: `gender`, `race`.

## Modele
Baseline: LogisticRegression scikit-learn.

## Performance baseline
- Accuracy: 0.658
- F1: 0.785
- ROC-AUC: 0.615

## Fairness baseline
- Disparate impact gender: 1.041
- DPD gender: -0.038

## Mitigations testees
- Reweighting
- Resampling

## Limites
Les donnees synthetiques servent a valider le pipeline. Un deploiement reel necessite validation domaine, revue juridique et monitoring continu.

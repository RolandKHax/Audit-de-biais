# Guide Complet : Audit de Biais d'un Modèle de Classification

## 1. FONDATIONS THÉORIQUES

### Comprendre les biais en Machine Learning

Avant toute chose, vous devez maîtriser la distinction entre trois types de biais : le **biais algorithmique** (provenant du modèle lui-même), le **biais de données** (échantillonnage non représentatif, sous-représentation de certains groupes), et le **biais de mesure** (labels biaisés, proxys imparfaits). Ces biais se manifestent souvent de manière intersectionnelle, c'est-à-dire qu'une femme d'origine minoritaire peut subir une discrimination amplifiée par rapport à chaque attribut pris séparément.

La notion d'**attributs protégés** (ou sensibles) est centrale. Il s'agit de caractéristiques comme le genre, l'origine ethnique, l'âge, la religion ou le handicap qui sont protégées par la loi dans de nombreux pays. Votre audit doit identifier ces attributs dans votre contexte spécifique, même lorsqu'ils sont implicites ou reconstruits via des proxys (par exemple, le code postal comme proxy de l'origine ethnique).

### Les définitions de la fairness

Il existe plusieurs définitions mathématiques de l'équité, et elles sont **mutuellement incompatibles** dans la plupart des cas (théorème d'impossibilité de la fairness). Vous devez comprendre cette tension fondamentale :

**Demographic Parity (Statistical Parity)** exige que la probabilité de recevoir une prédiction positive soit identique pour tous les groupes, indépendamment de leurs caractéristiques réelles. Formellement : P(Ŷ=1|A=a) = P(Ŷ=1|A=b) pour tous groupes a et b. Cette définition est appropriée quand on considère que les différences de taux de base entre groupes sont elles-mêmes le résultat de discriminations historiques.

**Equalized Odds** (égalité des opportunités) exige que le modèle ait les mêmes taux de vrais positifs et de faux positifs pour tous les groupes. Formellement : P(Ŷ=1|Y=y, A=a) = P(Ŷ=1|Y=y, A=b) pour tous y, a, b. Cette approche accepte des taux de sélection différents entre groupes si les taux de base sont différents, mais exige une performance prédictive équivalente.

**Equal Opportunity** est une version relaxée d'equalized odds qui ne considère que les vrais positifs (sensibilité). **Predictive Parity** exige que P(Y=1|Ŷ=1, A=a) = P(Y=1|Ŷ=1, A=b), c'est-à-dire que la précision positive soit identique entre groupes.

Vous devez choisir quelle(s) métrique(s) privilégier en fonction du contexte applicatif et des valeurs éthiques en jeu. Par exemple, dans un système de crédit, on pourrait privilégier equalized odds pour ne pas pénaliser des groupes qui ont historiquement moins accès au crédit.

## 2. MÉTHODOLOGIE D'AUDIT

### Phase de découverte et contextualisation

Commencez par documenter exhaustivement le contexte d'utilisation du modèle : quel est son objectif métier, quelles décisions sont prises sur la base de ses prédictions, quelles sont les conséquences pour les individus concernés, et quels sont les cadres légaux applicables (RGPD en Europe, Civil Rights Act aux États-Unis, etc.).

Identifiez les parties prenantes : data scientists, juristes, représentants des groupes potentiellement affectés, décideurs métier. Un audit de biais n'est pas qu'un exercice technique, c'est un processus participatif qui doit intégrer des perspectives diverses.

### Analyse exploratoire des données

Réalisez une analyse démographique approfondie de votre dataset. Calculez les distributions des attributs protégés dans vos données d'entraînement, de validation et de test. Vérifiez si certains groupes sont sous-représentés (moins de 5-10% peut poser problème). Analysez les corrélations entre attributs protégés et label cible : des différences importantes peuvent indiquer des biais historiques dans les données.

Examinez la qualité des labels pour chaque groupe. Les erreurs d'annotation sont-elles réparties uniformément ? Y a-t-il des annotateurs différents pour différents groupes ? Utilisez des techniques comme l'analyse inter-annotateur stratifiée par groupe.

Identifiez les **proxys cachés** : des variables apparemment neutres qui sont fortement corrélées avec des attributs protégés. Par exemple, dans un modèle de recrutement, "années d'expérience" peut être corrélé avec l'âge, "loisirs" avec le genre, "adresse" avec l'origine ethnique. Utilisez des analyses de corrélation, des modèles de régression ou des techniques comme le Fairness Through Unawareness test.

### Évaluation du modèle baseline

Avant toute intervention, établissez un baseline complet. Calculez les métriques de performance globales (accuracy, F1-score, AUC-ROC) mais aussi **stratifiées par groupe** pour chaque attribut protégé. Calculez toutes les métriques de fairness pertinentes : demographic parity difference/ratio, equalized odds difference, average odds difference, disparate impact ratio (ratio des taux de sélection, doit être > 0.8 selon la règle des 80%).

Utilisez des **matrices de confusion par groupe** pour visualiser où se situent les disparités. Créez des courbes ROC par groupe pour voir si certains groupes ont systématiquement des performances prédictives inférieures.

Appliquez des tests statistiques (chi-carré, test de permutation) pour déterminer si les différences observées sont statistiquement significatives. Avec de petits échantillons pour certains groupes, vous pourriez observer des différences dues au hasard.

### Analyse causale et intersectionnelle

Allez au-delà de l'analyse univariée. Examinez les interactions entre attributs protégés (intersectionnalité) : une femme noire peut être discriminée différemment qu'une femme blanche ou qu'un homme noir. Créez des sous-groupes intersectionnels et calculez les métriques pour ces sous-groupes, en gardant à l'esprit les limites statistiques avec des échantillons réduits.

Utilisez des techniques d'interprétabilité (SHAP, LIME) pour comprendre **pourquoi** le modèle discrimine. Quelles features contribuent le plus aux disparités ? Y a-t-il des interactions non-linéaires problématiques ?

## 3. TECHNIQUES D'ATTÉNUATION DES BIAIS

### Pré-traitement des données

Le **re-sampling** consiste à modifier la distribution des données d'entraînement. L'oversampling consiste à dupliquer (avec ou sans perturbations) des exemples de groupes sous-représentés. L'undersampling retire des exemples de groupes sur-représentés. Le SMOTE (Synthetic Minority Over-sampling Technique) génère des exemples synthétiques en interpolant entre exemples proches. Vous pouvez également faire du re-sampling stratifié par sous-groupe (groupe défavorisé ET label positif).

Le **re-weighing** assigne des poids différents aux exemples pendant l'entraînement. La formule standard est : w(x,y,s) = P(y)P(s) / P(y,s)P(s|y), qui compense les corrélations entre label et attribut protégé. Implémentez cela via le paramètre `sample_weight` dans scikit-learn ou `class_weight` dans Keras.

Le **re-labeling** (à utiliser avec précaution) modifie certains labels pour améliorer la fairness. Cela peut être justifié si on considère que certains labels historiques sont biaisés, mais pose des questions éthiques importantes.

Les **transformations d'apprentissage** comme Optimized Pre-Processing (algorithme de Calmon et al.) apprennent une transformation probabiliste optimale des features et labels qui maximise la fairness tout en préservant l'utilité.

### Modifications in-processing

L'**adversarial debiasing** est une technique sophistiquée inspirée des GANs. Vous entraînez simultanément deux réseaux : un prédicteur principal qui essaie de prédire Y, et un adversaire qui essaie de prédire l'attribut protégé A à partir des représentations internes du prédicteur. Le prédicteur est pénalisé s'il encode de l'information sur A, créant des représentations "aveugles" aux attributs protégés. La fonction de perte combine : L = L_prediction + λL_adversarial, où λ contrôle le trade-off fairness/accuracy.

Implémentez cela avec une architecture à deux têtes : une branche pour la tâche principale, une pour l'adversaire. Utilisez un gradient reversal layer pour que le prédicteur apprenne à "tromper" l'adversaire. Testez différentes valeurs de λ pour trouver le bon équilibre.

Les **contraintes de fairness régularisées** ajoutent directement un terme de pénalité à la fonction de perte. Par exemple, pour demographic parity : L = L_task + λ|P(Ŷ=1|A=0) - P(Ŷ=1|A=1)|. Pour equalized odds, pénalisez les différences de TPR et FPR entre groupes.

Le **fairness through unawareness** supprime l'attribut protégé des features. ATTENTION : cela ne suffit généralement pas car les proxys restent. C'est nécessaire mais pas suffisant.

Les **méthodes de threshold optimization** (comme Hardt et al.) apprennent des seuils de décision différents par groupe pour satisfaire des contraintes de fairness tout en maximisant l'accuracy globale.

### Post-traitement

Les techniques de **calibration par groupe** ajustent les seuils de décision ou les scores de probabilité après entraînement. L'algorithme de Platt scaling peut être appliqué séparément à chaque groupe. Le calibrage isotonique est une alternative non-paramétrique.

Le **reject option classification** définit une "zone d'incertitude" autour du seuil de décision. Pour les exemples dans cette zone provenant de groupes défavorisés, on favorise la décision positive. Pour les exemples dans cette zone provenant de groupes favorisés, on favorise la décision négative.

### Choix de la stratégie

Testez plusieurs approches et documentez leurs impacts sur la fairness ET la performance. Généralement, adversarial debiasing offre de bons résultats mais est plus complexe à implémenter et entraîner. Re-weighing est simple et efficace pour des biais modérés. Le post-traitement est utile quand vous ne pouvez pas réentraîner le modèle.

## 4. IMPLÉMENTATION TECHNIQUE

### Stack technologique recommandée

Note de cohérence pour ce dépôt : l'environnement exécutable actuel cible Python 3.14. Le workflow par défaut s'appuie donc sur scikit-learn, pandas, numpy, SHAP/LIME et des métriques internes dans `bias_audit/src/metrics.py`. AIF360, Fairlearn, TensorFlow et PyTorch restent des références utiles pour l'état de l'art, mais ne sont pas nécessaires pour lancer ce projet.

Utilisez **Python 3.8+** avec les bibliothèques suivantes :

**AIF360** (AI Fairness 360 d'IBM) est LA bibliothèque de référence pour les audits de fairness. Elle contient 70+ métriques de fairness, 10+ algorithmes de débiaisage (pre, in, post-processing), et des datasets de benchmark. Installez avec `pip install aif360`. Comprenez les classes BinaryLabelDataset, ClassificationMetric, et les différents algorithms.

**Fairlearn** (Microsoft) offre une interface plus simple et une bonne intégration avec scikit-learn. Elle excelle dans les techniques de post-processing (ThresholdOptimizer, ExponentiatedGradient). Plus accessible pour débuter.

**What-If Tool** (Google) est excellent pour l'exploration interactive et la visualisation des biais. Intégrez-le dans vos notebooks pour des démonstrations visuelles.

Pour l'adversarial debiasing, utilisez **TensorFlow 2.x** ou **PyTorch 1.x** pour implémenter les architectures custom. Regardez les implémentations de référence dans AIF360 comme point de départ.

Pour l'analyse et la visualisation, maîtrisez **pandas** (manipulation de données), **numpy** (calculs), **matplotlib/seaborn** (visualisations statiques), et **plotly** (visualisations interactives).

Pour l'interprétabilité, utilisez **SHAP** (SHapley Additive exPlanations) pour analyser les contributions des features, et **LIME** pour des explications locales.

### Architecture du projet

Organisez votre code en modules clairement séparés :

Un module `data_processing.py` contenant les fonctions de chargement, nettoyage, feature engineering, et préparation des datasets avec attributs protégés. Un module `metrics.py` avec toutes vos fonctions de calcul de métriques de fairness et performance, stratifiées par groupe. Un module `bias_mitigation.py` avec les implémentations de vos techniques de débiaisage. Un module `evaluation.py` pour orchestrer les évaluations complètes. Un module `visualization.py` pour toutes vos fonctions de plotting.

Créez des notebooks séparés pour : l'analyse exploratoire (`01_EDA.ipynb`), l'évaluation du baseline (`02_baseline_evaluation.ipynb`), les expérimentations de débiaisage (`03_debiasing_experiments.ipynb`), et l'analyse finale comparative (`04_final_analysis.ipynb`).

### Bonnes pratiques de code

Utilisez des **pipelines scikit-learn** pour encapsuler preprocessing, feature engineering, et model training. Cela garantit la reproductibilité et évite le data leakage.

Implémentez une **validation croisée stratifiée** qui maintient la distribution des attributs protégés dans chaque fold. Utilisez StratifiedGroupKFold si possible.

Versionnez vos données avec **DVC** (Data Version Control) et votre code avec Git. Trackez vos expériences avec **MLflow** ou **Weights & Biases** pour comparer les métriques de fairness et performance de différentes configurations.

Écrivez des **tests unitaires** pour vos fonctions de métriques (vérifiez avec des cas extrêmes connus). Documentez abondamment votre code avec des docstrings.

### Gestion des cas complexes

Si vous n'avez pas accès aux attributs protégés dans vos données (fréquent en Europe avec RGPD), vous avez plusieurs options : utiliser des proxys identifiables (avec prudence), faire de la fairness sans étiquettes (fairness through clustering), ou utiliser des techniques de fairness individuelle (similarité entre individus).

Pour les attributs protégés multi-classes (origine ethnique avec 10+ catégories), considérez des regroupements justifiés (par exemple, majorité vs minorités visibles) mais documentez ces choix et leurs limites.

Pour les datasets déséquilibrés, les métriques de fairness peuvent être instables. Utilisez le bootstrapping pour calculer des intervalles de confiance autour de vos métriques.

## 5. LIVRABLES ATTENDUS

### Rapport d'audit

Votre rapport doit suivre une structure rigoureuse :

Le **résumé exécutif** (1-2 pages) présente les findings principaux, les recommandations prioritaires, et les risques identifiés. Il doit être compréhensible par des non-experts.

L'**introduction** décrit le contexte du modèle audité, sa fonction métier, les enjeux éthiques et légaux, et les objectifs de l'audit.

La **méthodologie** détaille vos choix : datasets utilisés, métriques sélectionnées et leur justification, protocole d'évaluation, techniques testées. Expliquez pourquoi vous avez choisi demographic parity vs equalized odds par exemple.

Les **résultats quantitatifs** présentent vos findings de manière structurée. Pour le modèle baseline, montrez les métriques de fairness pour chaque attribut protégé, avec des visualisations (barplots de disparate impact, heatmaps de matrices de confusion par groupe, courbes ROC comparatives). Documentez les disparités statistiquement significatives.

Pour chaque technique de débiaisage testée, présentez l'impact sur les métriques de fairness ET de performance. Créez des tableaux comparatifs et des graphiques Pareto (fairness vs accuracy trade-off). Identifiez la ou les configurations qui offrent le meilleur compromis.

L'**analyse qualitative** interprète les résultats avec SHAP/LIME. Expliquez quelles features contribuent aux biais, identifiez les proxys problématiques, et discutez des patterns intersectionnels découverts.

Les **recommandations** doivent être actionnables et priorisées. Distinguez les actions court-terme (ajustement de seuils, re-weighing), moyen-terme (collecte de données plus équilibrées, réentraînement avec adversarial debiasing), et long-terme (refonte du processus métier, gouvernance des données). Pour chaque recommandation, spécifiez l'impact attendu, la difficulté de mise en œuvre, et les risques.

La **discussion des limites** est critique pour la crédibilité. Discutez des compromis fairness-accuracy, des attributs protégés non observés, des biais qui persistent malgré vos interventions, et des questions éthiques ouvertes (par exemple : est-il acceptable d'avoir des performances légèrement inférieures pour améliorer la fairness ?).

Incluez des **annexes techniques** avec les formules mathématiques complètes, les hyperparamètres testés, les résultats détaillés par sous-groupe, et le code reproductible.

### Scripts d'audit

Fournissez un **package Python installable** (avec setup.py) contenant vos modules. Incluez un README détaillé avec les instructions d'installation et d'utilisation.

Créez un **script principal d'audit** (`audit.py`) qui peut être exécuté en ligne de commande avec des arguments pour spécifier le modèle à auditer, les données, les attributs protégés, et les métriques à calculer. Exemple : `python audit.py --model model.pkl --data data.csv --protected-attrs gender,race --metrics demographic-parity,equalized-odds --output report.html`.

Le script doit générer automatiquement : un fichier JSON avec toutes les métriques calculées, des graphiques sauvegardés en haute résolution, et un rapport HTML interactif avec des visualisations exploitables.

Incluez un **notebook de démonstration** (`demo.ipynb`) qui guide à travers un audit complet sur un dataset de référence (Adult Income, COMPAS, ou German Credit). Ce notebook doit être didactique avec des explications de chaque étape.

Fournissez des **scripts de débiaisage** réutilisables (`debias.py`) qui appliquent les techniques que vous avez trouvées efficaces. Ces scripts doivent être modulaires et configurables.

### Recommandations stratégiques

Au-delà de l'aspect technique, formulez des recommandations sur la **gouvernance de l'IA** : qui doit être responsable du monitoring continu des biais, à quelle fréquence ré-auditer le modèle, comment documenter les décisions (model cards, datasheets for datasets), comment impliquer les parties prenantes.

Proposez un **framework de monitoring en production** : quelles métriques tracker en temps réel, quels seuils d'alerte définir, comment détecter un drift démographique ou une dégradation de la fairness.

Discutez des **considérations légales** : RGPD (droit à l'explication, profilage automatisé), AI Act européen, jurisprudence pertinente sur la discrimination algorithmique.

## 6. CRITÈRES D'EXCELLENCE

Pour obtenir un score maximal, votre projet doit démontrer :

**Rigueur méthodologique** : protocole d'évaluation robuste avec validation croisée stratifiée, tests statistiques de significativité, intervalles de confiance sur les métriques, analyse de sensibilité des hyperparamètres.

**Profondeur d'analyse** : ne vous contentez pas de calculer des métriques, interprétez-les, cherchez les causes racines avec des analyses causales, explorez l'intersectionnalité, utilisez l'interprétabilité pour comprendre les mécanismes de discrimination.

**Comparaison exhaustive** : testez au minimum 3-4 techniques de débiaisage différentes (au moins une de chaque catégorie : pre, in, post-processing), comparez-les sur plusieurs métriques, documentez les trade-offs.

**Qualité des livrables** : rapport professionnel avec visualisations de qualité publication, code propre et documenté avec des tests, scripts réutilisables et généralisables, documentation utilisateur complète.

**Conscience éthique** : discussion nuancée des dilemmes éthiques, reconnaissance des limites de l'approche technique, considération des impacts sociaux, recommandations qui vont au-delà du technique vers la gouvernance.

**Reproductibilité** : code versionné, environnement reproductible (requirements.txt ou environment.yml), données versionnées ou procédure de téléchargement claire, notebooks exécutables de bout en bout, seed fixé pour l'aléatoire.

## 7. RESSOURCES ET APPROFONDISSEMENTS

Lisez les papers fondamentaux : "Fairness Through Awareness" (Dwork et al., 2012), "Equality of Opportunity in Supervised Learning" (Hardt et al., 2016), "Fair prediction with disparate impact" (Zafar et al., 2017), "Mitigating Unwanted Biases with Adversarial Learning" (Zhang et al., 2018).

Consultez les guidelines officiels : "A Framework for Understanding Sources of Harm throughout the Machine Learning Life Cycle" (Suresh & Guttag, 2021), les recommandations de l'OECD sur l'IA, le guide de la CNIL sur l'IA.

Explorez les ressources pratiques : le cours "Fairness in Machine Learning" de Moritz Hardt, les tutoriels AIF360 et Fairlearn, les case studies de Partnership on AI.

Pratiquez sur des benchmarks reconnus : Adult Income Dataset, COMPAS Recidivism Risk Scores, German Credit Data, Bank Marketing Dataset. Comparez vos résultats avec ceux publiés dans la littérature.

---

Ce projet est une opportunité de démontrer non seulement vos compétences techniques en ML, mais aussi votre maturité éthique et votre capacité à naviguer les compromis complexes entre performance, fairness, et viabilité opérationnelle. Abordez-le avec rigueur scientifique et conscience sociale. Bonne chance !

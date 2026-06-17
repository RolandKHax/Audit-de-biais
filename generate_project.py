"""
Générateur de structure de projet pour l'Audit de Biais en ML
Exécutez ce script pour créer tous les fichiers nécessaires au projet.

Usage: python generate_project.py
"""

import os
from pathlib import Path

def create_directory_structure():
    """Crée la structure de dossiers du projet"""
    directories = [
        'bias_audit',
        'bias_audit/data',
        'bias_audit/data/raw',
        'bias_audit/data/processed',
        'bias_audit/src',
        'bias_audit/notebooks',
        'bias_audit/reports',
        'bias_audit/reports/figures',
        'bias_audit/tests',
        'bias_audit/configs',
        'bias_audit/models',
        'bias_audit/outputs',
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Créé: {directory}")

def create_requirements():
    """Crée le fichier requirements.txt"""
    content = """# Python support
# This dependency set targets Python 3.14.
# Fairness metrics are implemented in src/metrics.py. AIF360, Fairlearn,
# TensorFlow and pinned old PyTorch builds were removed because they force
# older binary dependencies or do not provide Python 3.14-compatible wheels.

# Core ML Libraries
numpy==2.4.6
pandas==2.3.3
scikit-learn==1.8.0
scipy==1.17.1

# Visualization
matplotlib==3.10.9
seaborn==0.13.2
plotly==6.7.0

# Interpretability
shap==0.51.0
lime==0.2.0.1

# Experiment Tracking
mlflow==3.12.0
wandb==0.27.0

# Utilities
tqdm==4.67.3
joblib==1.5.3
pyyaml==6.0.3
python-dotenv==1.2.2

# Jupyter
jupyter==1.1.1
ipywidgets==8.1.8
notebook==7.5.6

# Testing
pytest==9.0.3
pytest-cov==7.1.0

# Documentation
sphinx==9.1.0
sphinx-rtd-theme==3.1.0
"""
    
    with open('bias_audit/requirements.txt', 'w') as f:
        f.write(content)
    print("✓ Créé: requirements.txt")

def create_data_processing():
    """Crée le module de traitement de données"""
    content = """\"\"\"
Module de traitement et préparation des données pour l'audit de biais.
\"\"\"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    \"\"\"Classe pour le traitement et la préparation des données\"\"\"
    
    def __init__(self, protected_attributes: List[str], label_name: str):
        \"\"\"
        Args:
            protected_attributes: Liste des attributs protégés
            label_name: Nom de la colonne cible
        \"\"\"
        self.protected_attributes = protected_attributes
        self.label_name = label_name
        self.encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        \"\"\"Charge les données depuis un fichier CSV\"\"\"
        try:
            df = pd.read_csv(filepath)
            print(f"Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
            return df
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            return None
    
    def explore_demographics(self, df: pd.DataFrame) -> Dict:
        \"\"\"Analyse démographique des attributs protégés\"\"\"
        demographics = {}
        
        for attr in self.protected_attributes:
            if attr in df.columns:
                counts = df[attr].value_counts()
                proportions = df[attr].value_counts(normalize=True)
                demographics[attr] = {
                    'counts': counts.to_dict(),
                    'proportions': proportions.to_dict(),
                    'unique_values': df[attr].nunique()
                }
                
        # Distribution des labels
        if self.label_name in df.columns:
            demographics['label'] = {
                'counts': df[self.label_name].value_counts().to_dict(),
                'proportions': df[self.label_name].value_counts(normalize=True).to_dict()
            }
            
        # Analyse intersectionnelle
        if len(self.protected_attributes) >= 2:
            demographics['intersectional'] = {}
            for i, attr1 in enumerate(self.protected_attributes):
                for attr2 in self.protected_attributes[i+1:]:
                    key = f"{attr1}_{attr2}"
                    crosstab = pd.crosstab(df[attr1], df[attr2])
                    demographics['intersectional'][key] = crosstab.to_dict()
        
        return demographics
    
    def check_data_quality(self, df: pd.DataFrame) -> Dict:
        \"\"\"Vérifie la qualité des données\"\"\"
        quality_report = {
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'dtypes': df.dtypes.astype(str).to_dict()
        }
        
        # Détection d'outliers pour les colonnes numériques
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_condition = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
            outliers[col] = outlier_condition.sum()
        
        quality_report['outliers'] = outliers
        
        return quality_report
    
    def encode_categorical(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        \"\"\"Encode les variables catégorielles\"\"\"
        df_encoded = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=['object']).columns
        
        for col in columns:
            if col in df.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    df_encoded[col] = self.encoders[col].fit_transform(df[col].astype(str))
                else:
                    df_encoded[col] = self.encoders[col].transform(df[col].astype(str))
        
        return df_encoded
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple:
        \"\"\"Standardise les features numériques\"\"\"
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        
        X_train_scaled = X_train.copy()
        X_train_scaled[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        
        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def create_fairness_inputs(self, df: pd.DataFrame) -> Dict[str, Any]:
        \"\"\"Prépare les entrées utiles aux métriques d'audit de fairness.\"\"\"
        missing_columns = [
            col for col in [self.label_name, *self.protected_attributes]
            if col not in df.columns
        ]
        if missing_columns:
            raise ValueError(f"Colonnes manquantes: {missing_columns}")

        X = df.drop(columns=[self.label_name])
        y = df[self.label_name]
        sensitive_features = df[self.protected_attributes]

        for attr in self.protected_attributes:
            if df[attr].isna().any():
                raise ValueError(f"L'attribut protégé '{attr}' contient des valeurs manquantes")

        return {
            'X': X,
            'y': y,
            'sensitive_features': sensitive_features,
        }
    
    def identify_proxies(self, df: pd.DataFrame, threshold: float = 0.5) -> Dict:
        \"\"\"Identifie les proxys potentiels des attributs protégés\"\"\"
        from scipy.stats import chi2_contingency
        from scipy.stats import pearsonr
        
        proxies = {}
        
        for protected_attr in self.protected_attributes:
            if protected_attr not in df.columns:
                continue
                
            proxies[protected_attr] = []
            
            for col in df.columns:
                if col == protected_attr or col in self.protected_attributes or col == self.label_name:
                    continue
                
                # Pour variables catégorielles: test chi-2
                if df[col].dtype == 'object' or df[col].nunique() < 10:
                    try:
                        contingency = pd.crosstab(df[protected_attr], df[col])
                        chi2, p_value, dof, expected = chi2_contingency(contingency)
                        
                        # Cramér's V pour mesure d'association
                        n = contingency.sum().sum()
                        cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
                        
                        if cramers_v > threshold:
                            proxies[protected_attr].append({
                                'feature': col,
                                'correlation': cramers_v,
                                'p_value': p_value,
                                'type': 'categorical'
                            })
                    except:
                        pass
                
                # Pour variables numériques: corrélation de Pearson
                else:
                    try:
                        # Encoder l'attribut protégé si nécessaire
                        if df[protected_attr].dtype == 'object':
                            protected_encoded = LabelEncoder().fit_transform(df[protected_attr])
                        else:
                            protected_encoded = df[protected_attr]
                        
                        corr, p_value = pearsonr(protected_encoded, df[col])
                        
                        if abs(corr) > threshold:
                            proxies[protected_attr].append({
                                'feature': col,
                                'correlation': abs(corr),
                                'p_value': p_value,
                                'type': 'numeric'
                            })
                    except:
                        pass
            
            # Trier par corrélation décroissante
            proxies[protected_attr] = sorted(
                proxies[protected_attr], 
                key=lambda x: x['correlation'], 
                reverse=True
            )
        
        return proxies
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2, 
                  random_state: int = 42, stratify: bool = True) -> Tuple:
        \"\"\"Divise les données en train/test avec stratification sur attributs protégés\"\"\"
        
        X = df.drop(columns=[self.label_name])
        y = df[self.label_name]
        
        if stratify and len(self.protected_attributes) > 0:
            # Créer une colonne de stratification combinée
            stratify_col = df[self.protected_attributes[0]].astype(str)
            for attr in self.protected_attributes[1:]:
                stratify_col = stratify_col + "_" + df[attr].astype(str)
            stratify_col = stratify_col + "_" + y.astype(str)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state,
                stratify=stratify_col
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        return X_train, X_test, y_train, y_test


def generate_sample_data(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    \"\"\"Génère un dataset synthétique avec biais pour démonstration\"\"\"
    np.random.seed(random_state)
    
    # Attributs protégés
    gender = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.6, 0.4])
    race = np.random.choice(['White', 'Black', 'Asian', 'Hispanic'], 
                           size=n_samples, p=[0.6, 0.2, 0.1, 0.1])
    age = np.random.randint(18, 70, size=n_samples)
    
    # Features corrélées avec attributs protégés (proxies)
    education_levels = ['High School', 'Bachelor', 'Master', 'PhD']
    education = []
    for g in gender:
        if g == 'Male':
            education.append(np.random.choice(education_levels, p=[0.2, 0.3, 0.3, 0.2]))
        else:
            education.append(np.random.choice(education_levels, p=[0.3, 0.4, 0.2, 0.1]))
    
    # Features neutres
    years_experience = np.random.randint(0, 30, size=n_samples)
    test_score = np.random.normal(75, 15, size=n_samples)
    
    # Label avec biais intentionnel
    label = []
    for i in range(n_samples):
        base_prob = 0.3
        
        # Biais de genre
        if gender[i] == 'Male':
            base_prob += 0.2
        
        # Biais racial
        if race[i] == 'White':
            base_prob += 0.15
        
        # Facteurs légitimes
        if education[i] in ['Master', 'PhD']:
            base_prob += 0.1
        if years_experience[i] > 10:
            base_prob += 0.1
        if test_score[i] > 80:
            base_prob += 0.1
        
        label.append(1 if np.random.random() < base_prob else 0)
    
    df = pd.DataFrame({
        'gender': gender,
        'race': race,
        'age': age,
        'education': education,
        'years_experience': years_experience,
        'test_score': test_score,
        'label': label
    })
    
    return df


if __name__ == "__main__":
    # Test du module
    print("Test du module de traitement de données...")
    
    # Générer des données de test
    df = generate_sample_data(n_samples=5000)
    print(f"\\nDataset généré: {df.shape}")
    
    # Initialiser le processor
    processor = DataProcessor(
        protected_attributes=['gender', 'race'],
        label_name='label'
    )
    
    # Analyse démographique
    demographics = processor.explore_demographics(df)
    print(f"\\nAnalyse démographique:")
    for attr, stats in demographics.items():
        print(f"  {attr}: {stats}")
    
    # Vérification de qualité
    quality = processor.check_data_quality(df)
    print(f"\\nQualité des données:")
    print(f"  Valeurs manquantes: {quality['missing_values']}")
    print(f"  Doublons: {quality['duplicates']}")
    
    # Identification des proxies
    proxies = processor.identify_proxies(df, threshold=0.3)
    print(f"\\nProxies identifiés:")
    for attr, proxy_list in proxies.items():
        print(f"  {attr}: {len(proxy_list)} proxies potentiels")
        for proxy in proxy_list[:3]:
            print(f"    - {proxy['feature']}: {proxy['correlation']:.3f}")
    
    print("\\n✓ Module testé avec succès!")
"""
    
    with open('bias_audit/src/data_processing.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✓ Créé: src/data_processing.py")

def create_metrics_module():
    """Crée le module de calcul des métriques"""
    content = """\"\"\"
Module de calcul des métriques de fairness et de performance.
\"\"\"

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class FairnessMetrics:
    \"\"\"Calculateur de métriques de fairness\"\"\"
    
    def __init__(self, protected_attribute: str, privileged_value, unprivileged_value):
        \"\"\"
        Args:
            protected_attribute: Nom de l'attribut protégé
            privileged_value: Valeur du groupe privilégié
            unprivileged_value: Valeur du groupe défavorisé
        \"\"\"
        self.protected_attribute = protected_attribute
        self.privileged_value = privileged_value
        self.unprivileged_value = unprivileged_value
    
    def demographic_parity_difference(self, y_pred: np.ndarray, 
                                     sensitive_features: np.ndarray) -> float:
        \"\"\"
        Calcule la différence de parité démographique.
        
        DPD = P(Ŷ=1|A=privileged) - P(Ŷ=1|A=unprivileged)
        Idéalement proche de 0. Valeurs acceptables: [-0.1, 0.1]
        \"\"\"
        priv_mask = sensitive_features == self.privileged_value
        unpriv_mask = sensitive_features == self.unprivileged_value
        
        priv_rate = np.mean(y_pred[priv_mask])
        unpriv_rate = np.mean(y_pred[unpriv_mask])
        
        return priv_rate - unpriv_rate
    
    def demographic_parity_ratio(self, y_pred: np.ndarray,
                                sensitive_features: np.ndarray) -> float:
        \"\"\"
        Calcule le ratio de parité démographique.
        
        DPR = P(Ŷ=1|A=unprivileged) / P(Ŷ=1|A=privileged)
        Idéalement proche de 1. Valeurs acceptables: [0.8, 1.25]
        \"\"\"
        priv_mask = sensitive_features == self.privileged_value
        unpriv_mask = sensitive_features == self.unprivileged_value
        
        priv_rate = np.mean(y_pred[priv_mask])
        unpriv_rate = np.mean(y_pred[unpriv_mask])
        
        if priv_rate == 0:
            return np.inf
        
        return unpriv_rate / priv_rate
    
    def disparate_impact(self, y_pred: np.ndarray,
                        sensitive_features: np.ndarray) -> float:
        \"\"\"
        Calcule le disparate impact (ratio des taux de sélection).
        
        DI = P(Ŷ=1|A=unprivileged) / P(Ŷ=1|A=privileged)
        Règle des 80%: DI devrait être >= 0.8
        \"\"\"
        return self.demographic_parity_ratio(y_pred, sensitive_features)
    
    def equalized_odds_difference(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  sensitive_features: np.ndarray) -> Dict[str, float]:
        \"\"\"
        Calcule la différence d'equalized odds.
        
        Mesure la différence de TPR et FPR entre groupes.
        EOD_TPR = TPR(privileged) - TPR(unprivileged)
        EOD_FPR = FPR(privileged) - FPR(unprivileged)
        \"\"\"
        priv_mask = sensitive_features == self.privileged_value
        unpriv_mask = sensitive_features == self.unprivileged_value
        
        # True Positive Rate
        tpr_priv = self._tpr(y_true[priv_mask], y_pred[priv_mask])
        tpr_unpriv = self._tpr(y_true[unpriv_mask], y_pred[unpriv_mask])
        
        # False Positive Rate
        fpr_priv = self._fpr(y_true[priv_mask], y_pred[priv_mask])
        fpr_unpriv = self._fpr(y_true[unpriv_mask], y_pred[unpriv_mask])
        
        return {
            'tpr_difference': tpr_priv - tpr_unpriv,
            'fpr_difference': fpr_priv - fpr_unpriv,
            'average_odds_difference': 0.5 * ((tpr_priv - tpr_unpriv) + (fpr_priv - fpr_unpriv))
        }
    
    def equal_opportunity_difference(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     sensitive_features: np.ndarray) -> float:
        \"\"\"
        Calcule la différence d'égalité des opportunités.
        
        EOppD = TPR(privileged) - TPR(unprivileged)
        Se concentre uniquement sur les vrais positifs.
        \"\"\"
        priv_mask = sensitive_features == self.privileged_value
        unpriv_mask = sensitive_features == self.unprivileged_value
        
        tpr_priv = self._tpr(y_true[priv_mask], y_pred[priv_mask])
        tpr_unpriv = self._tpr(y_true[unpriv_mask], y_pred[unpriv_mask])
        
        return tpr_priv - tpr_unpriv
    
    def predictive_parity_difference(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     sensitive_features: np.ndarray) -> float:
        \"\"\"
        Calcule la différence de parité prédictive.
        
        PPD = Precision(privileged) - Precision(unprivileged)
        \"\"\"
        priv_mask = sensitive_features == self.privileged_value
        unpriv_mask = sensitive_features == self.unprivileged_value
        
        # Précision pour chaque groupe
        prec_priv = precision_score(y_true[priv_mask], y_pred[priv_mask], zero_division=0)
        prec_unpriv = precision_score(y_true[unpriv_mask], y_pred[unpriv_mask], zero_division=0)
        
        return prec_priv - prec_unpriv
    
    def statistical_parity_difference(self, y_pred: np.ndarray,
                                     sensitive_features: np.ndarray) -> float:
        \"\"\"Alias pour demographic_parity_difference\"\"\"
        return self.demographic_parity_difference(y_pred, sensitive_features)
    
    def compute_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                           sensitive_features: np.ndarray, y_scores: np.ndarray = None) -> Dict:
        \"\"\"Calcule toutes les métriques de fairness\"\"\"
        
        metrics = {}
        
        # Métriques basées uniquement sur les prédictions
        metrics['demographic_parity_difference'] = self.demographic_parity_difference(
            y_pred, sensitive_features
        )
        metrics['demographic_parity_ratio'] = self.demographic_parity_ratio(
            y_pred, sensitive_features
        )
        metrics['disparate_impact'] = self.disparate_impact(y_pred, sensitive_features)
        
        # Métriques nécessitant les vraies valeurs
        eod = self.equalized_odds_difference(y_true, y_pred, sensitive_features)
        metrics.update(eod)
        
        metrics['equal_opportunity_difference'] = self.equal_opportunity_difference(
            y_true, y_pred, sensitive_features
        )
        
        metrics['predictive_parity_difference'] = self.predictive_parity_difference(
            y_true, y_pred, sensitive_features
        )
        
        # Interprétation automatique
        metrics['interpretation'] = self._interpret_metrics(metrics)
        
        return metrics
    
    def _tpr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        \"\"\"Calcule le True Positive Rate\"\"\"
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape[0] < 2 or cm.shape[1] < 2 or (cm[1, 0] + cm[1, 1]) == 0:
            return 0.0
        return cm[1, 1] / (cm[1, 0] + cm[1, 1])
    
    def _fpr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        \"\"\"Calcule le False Positive Rate\"\"\"
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape[0] < 2 or cm.shape[1] < 2 or (cm[0, 0] + cm[0, 1]) == 0:
            return 0.0
        return cm[0, 1] / (cm[0, 0] + cm[0, 1])
    
    def _interpret_metrics(self, metrics: Dict) -> Dict[str, str]:
        \"\"\"Interprète les métriques calculées\"\"\"
        interpretation = {}
        
        # Disparate Impact
        di = metrics['disparate_impact']
        if di >= 0.8 and di <= 1.25:
            interpretation['disparate_impact'] = "✓ Acceptable (règle des 80%)"
        elif di < 0.8:
            interpretation['disparate_impact'] = "✗ Discrimination possible (groupe défavorisé sous-sélectionné)"
        else:
            interpretation['disparate_impact'] = "⚠ Discrimination inverse possible"
        
        # Demographic Parity
        dpd = abs(metrics['demographic_parity_difference'])
        if dpd <= 0.1:
            interpretation['demographic_parity'] = "✓ Bonne parité démographique"
        elif dpd <= 0.2:
            interpretation['demographic_parity'] = "⚠ Parité démographique modérée"
        else:
            interpretation['demographic_parity'] = "✗ Parité démographique faible"
        
        # Equalized Odds
        avg_odds = abs(metrics['average_odds_difference'])
        if avg_odds <= 0.1:
            interpretation['equalized_odds'] = "✓ Bonnes odds égalisées"
        elif avg_odds <= 0.2:
            interpretation['equalized_odds'] = "⚠ Odds égalisées modérées"
        else:
            interpretation['equalized_odds'] = "✗ Odds égalisées faibles"
        
        return interpretation


class PerformanceMetrics:
    \"\"\"Calculateur de métriques de performance\"\"\"
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                       y_scores: np.ndarray = None) -> Dict:
        \"\"\"Calcule les métriques de performance classiques\"\"\"
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_scores is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_scores)
            except:
                metrics['roc_auc'] = None
        
        # Matrice de confusion
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = tn
            metrics['false_positives'] = fp
            metrics['false_negatives'] = fn
            metrics['true_positives'] = tp
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return metrics
    
    @staticmethod
    def compute_stratified_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                   sensitive_features: np.ndarray,
                                   group_values: List,
                                   y_scores: np.ndarray = None) -> Dict:
        \"\"\"Calcule les métriques stratifiées par groupe\"\"\"
        
        stratified = {}
        
        for group_value in group_values:
            mask = sensitive_features == group_value
            
            if mask.sum() == 0:
                continue
            
            group_scores = y_scores[mask] if y_scores is not None else None
            
            stratified[str(group_value)] = PerformanceMetrics.compute_metrics(
                y_true[mask], y_pred[mask], group_scores
            )
            
            stratified[str(group_value)]['sample_size'] = mask.sum()
        
        return stratified


def compare_fairness_performance(baseline_metrics: Dict, debiased_metrics: Dict) -> pd.DataFrame:
    \"\"\"Compare les métriques entre modèle baseline et débiaisé\"\"\"
    
    comparison = []
    
    for metric_name in baseline_metrics.keys():
        if metric_name == 'interpretation' or metric_name == 'confusion_matrix':
            continue
            
        baseline_val = baseline_metrics[metric_name]
        debiased_val = debiased_metrics.get(metric_name, None)
        
        if debiased_val is not None and isinstance(baseline_val, (int, float)):
            change = debiased_val - baseline_val
            pct_change = (change / abs(baseline_val)) * 100 if baseline_val != 0 else 0
            
            comparison.append({
                'Metric': metric_name,
                'Baseline': f"{baseline_val:.4f}",
                'Debiased': f"{debiased_val:.4f}",
                'Change': f"{change:+.4f}",
                'Change (%)': f"{pct_change:+.2f}%"
            })
    
    return pd.DataFrame(comparison)


if __name__ == "__main__":
    # Test du module
    print("Test du module de métriques...")
    
    # Données synthétiques
    np.random.seed(42)
    n = 1000
    
    y_true = np.random.randint(0, 2, n)
    y_pred = np.random.randint(0, 2, n)
    sensitive = np.random.choice(['A', 'B'], n, p=[0.6, 0.4])
    y_scores = np.random.random(n)
    
    # Test FairnessMetrics
    fm = FairnessMetrics(protected_attribute='group', 
                        privileged_value='A',
                        unprivileged_value='B')
    
    fairness_metrics = fm.compute_all_metrics(y_true, y_pred, sensitive, y_scores)
    
    print("\\nMétriques de Fairness:")
    for metric, value in fairness_metrics.items():
        if metric != 'interpretation':
            print(f"  {metric}: {value}")
    
    print("\\nInterprétation:")
    for metric, interp in fairness_metrics['interpretation'].items():
        print(f"  {metric}: {interp}")
    
    # Test PerformanceMetrics
    perf_metrics = PerformanceMetrics.compute_metrics(y_true, y_pred, y_scores)
    
    print("\\nMétriques de Performance:")
    for metric, value in perf_metrics.items():
        if metric != 'confusion_matrix':
            print(f"  {metric}: {value}")
    
    # Test métriques stratifiées
    stratified = PerformanceMetrics.compute_stratified_metrics(
        y_true, y_pred, sensitive, ['A', 'B'], y_scores
    )
    
    print("\\nMétriques Stratifiées:")
    for group, metrics in stratified.items():
        print(f"  Groupe {group}:")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    F1-Score: {metrics['f1_score']:.4f}")
    
    print("\\n✓ Module testé avec succès!")
"""
    
    with open('bias_audit/src/metrics.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("✓ Créé: src/metrics.py")

def create_main_files():
    """Crée les fichiers principaux du projet"""
    
    # README.md
    readme = """# Audit de biais - modele de classification

Projet executable d'audit et d'attenuation des biais pour un modele de classification binaire.

## Installation

```bash
cd bias_audit
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip check
```

Important: la commande correcte est `pip install -r requirements.txt`.

## Execution rapide

```bash
python -c "from src.data_processing import generate_sample_data; generate_sample_data(5000).to_csv('data/processed/dataset.csv', index=False)"

python -m src.audit_main \\
  --data data/processed/dataset.csv \\
  --protected-attrs gender race \\
  --label label \\
  --debiasing reweighting resampling \\
  --output reports/audit_report.html \\
  --output-dir reports/figures
```

## Techniques supportees par defaut

- Reweighting
- Resampling

Les approches adversariales et certains post-traitements restent des extensions possibles, mais elles ne sont pas activees dans l'environnement Python 3.14 par defaut.

## Tests

```bash
python -m pytest tests -q
python -m compileall -q src tests
```

## Rapport LaTeX

```bash
cd reports/latex
pdflatex rapport_audit_biais.tex
bibtex rapport_audit_biais
pdflatex rapport_audit_biais.tex
pdflatex rapport_audit_biais.tex
```
"""
    
    with open('bias_audit/README.md', 'w', encoding='utf-8') as f:
        f.write(readme)
    print("✓ Créé: README.md")
    
    # .gitignore
    gitignore = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints

# Environment
venv/
ENV/
env/
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data
data/raw/*
!data/raw/.gitkeep
data/processed/*
!data/processed/.gitkeep

# Models
models/*
!models/.gitkeep

# Outputs
outputs/*
!outputs/.gitkeep

# Reports
reports/figures/*
!reports/figures/.gitkeep

# OS
.DS_Store
Thumbs.db

# MLflow
mlruns/

# Weights & Biases
wandb/

# LaTeX
*.aux
*.log
*.out
*.toc
*.bbl
*.blg
*.synctex.gz
"""
    
    with open('bias_audit/.gitignore', 'w', encoding='utf-8') as f:
        f.write(gitignore)
    print("✓ Créé: .gitignore")
    
    # __init__.py files
    for init_file in ['bias_audit/src/__init__.py', 'bias_audit/tests/__init__.py']:
        with open(init_file, 'w') as f:
            f.write('"""Package initialization"""\\n')
    print("✓ Créé: __init__.py files")
    
    # .gitkeep files
    for gitkeep in ['data/raw/.gitkeep', 'data/processed/.gitkeep', 
                    'models/.gitkeep', 'outputs/.gitkeep', 'reports/figures/.gitkeep']:
        with open(f'bias_audit/{gitkeep}', 'w') as f:
            f.write('')
    print("✓ Créé: .gitkeep files")

def main():
    """Fonction principale pour générer le projet"""
    print("=" * 60)
    print("GÉNÉRATEUR DE PROJET - AUDIT DE BIAIS ML")
    print("=" * 60)
    print()
    
    create_directory_structure()
    print()
    
    create_requirements()
    create_data_processing()
    create_metrics_module()
    create_main_files()
    
    print()
    print("=" * 60)
    print("✓ PROJET CRÉÉ AVEC SUCCÈS!")
    print("=" * 60)
    print()
    print("Prochaines étapes:")
    print("1. cd bias_audit")
    print("2. python -m venv venv")
    print("3. source venv/bin/activate  (ou venv\\Scripts\\activate sur Windows)")
    print("4. python -m pip install --upgrade pip setuptools wheel")
    print("5. python -m pip install -r requirements.txt")
    print("   Attention: utiliser -r, pas '-'")
    print("6. jupyter notebook notebooks/")
    print()
    print("Les fichiers suivants seront créés ensuite:")
    print("  - src/bias_mitigation.py (techniques de débiaisage)")
    print("  - src/visualization.py (visualisations)")
    print("  - src/audit_main.py (script principal)")
    print("  - notebooks/*.ipynb (4 notebooks d'analyse)")
    print("  - reports/latex/rapport_audit_biais.tex")
    print()

if __name__ == "__main__":
    main()

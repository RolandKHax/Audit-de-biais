"""
Module de traitement et préparation des données pour l'audit de biais.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')


class DataProcessor:
    """Classe pour le traitement et la préparation des données"""
    
    def __init__(self, protected_attributes: List[str], label_name: str):
        """
        Args:
            protected_attributes: Liste des attributs protégés
            label_name: Nom de la colonne cible
        """
        self.protected_attributes = protected_attributes
        self.label_name = label_name
        self.encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Charge les données depuis un fichier CSV"""
        try:
            df = pd.read_csv(filepath)
            print(f"Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes")
            return df
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            return None
    
    def explore_demographics(self, df: pd.DataFrame) -> Dict:
        """Analyse démographique des attributs protégés"""
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
        """Vérifie la qualité des données"""
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
        """Encode les variables catégorielles"""
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
        """Standardise les features numériques"""
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        
        X_train_scaled = X_train.copy()
        X_train_scaled[numeric_cols] = self.scaler.fit_transform(X_train[numeric_cols])
        
        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[numeric_cols] = self.scaler.transform(X_test[numeric_cols])
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def create_fairness_inputs(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prépare les entrées utiles aux métriques d'audit de fairness."""
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
        """Identifie les proxys potentiels des attributs protégés"""
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
        """Divise les données en train/test avec stratification sur attributs protégés"""
        
        X = df.drop(columns=[self.label_name])
        y = df[self.label_name]
        
        if stratify and len(self.protected_attributes) > 0:
            # Créer une colonne de stratification combinée
            stratify_col = df[self.protected_attributes[0]].astype(str)
            for attr in self.protected_attributes[1:]:
                stratify_col = stratify_col + "_" + df[attr].astype(str)
            stratify_col = stratify_col + "_" + y.astype(str)
            
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state,
                    stratify=stratify_col
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state,
                    stratify=y
                )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        return X_train, X_test, y_train, y_test


def prepare_compas_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise un export COMPAS ProPublica pour l'audit.

    La fonction conserve les colonnes utiles, applique les filtres usuels de
    l'analyse ProPublica quand les colonnes existent, et garde `race` et `sex`
    comme attributs sensibles exploitables pour l'évaluation.
    """
    compas = df.copy()

    if 'days_b_screening_arrest' in compas.columns:
        compas = compas[
            compas['days_b_screening_arrest'].between(-30, 30, inclusive='both')
        ]
    if 'is_recid' in compas.columns:
        compas = compas[compas['is_recid'] != -1]
    if 'c_charge_degree' in compas.columns:
        compas = compas[compas['c_charge_degree'] != 'O']
    if 'score_text' in compas.columns:
        compas = compas[compas['score_text'] != 'N/A']

    preferred_columns = [
        'age',
        'age_cat',
        'race',
        'sex',
        'priors_count',
        'c_charge_degree',
        'juv_fel_count',
        'juv_misd_count',
        'juv_other_count',
        'decile_score',
        'score_text',
        'two_year_recid',
    ]
    available = [col for col in preferred_columns if col in compas.columns]
    if 'two_year_recid' not in available:
        raise ValueError("Le preset COMPAS exige la colonne cible 'two_year_recid'.")

    compas = compas[available].dropna().reset_index(drop=True)
    compas['two_year_recid'] = compas['two_year_recid'].astype(int)
    return compas


def generate_sample_data(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """Génère un dataset synthétique avec biais pour démonstration"""
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
    print(f"\nDataset généré: {df.shape}")
    
    # Initialiser le processor
    processor = DataProcessor(
        protected_attributes=['gender', 'race'],
        label_name='label'
    )
    
    # Analyse démographique
    demographics = processor.explore_demographics(df)
    print(f"\nAnalyse démographique:")
    for attr, stats in demographics.items():
        print(f"  {attr}: {stats}")
    
    # Vérification de qualité
    quality = processor.check_data_quality(df)
    print(f"\nQualité des données:")
    print(f"  Valeurs manquantes: {quality['missing_values']}")
    print(f"  Doublons: {quality['duplicates']}")
    
    # Identification des proxies
    proxies = processor.identify_proxies(df, threshold=0.3)
    print(f"\nProxies identifiés:")
    for attr, proxy_list in proxies.items():
        print(f"  {attr}: {len(proxy_list)} proxies potentiels")
        for proxy in proxy_list[:3]:
            print(f"    - {proxy['feature']}: {proxy['correlation']:.3f}")
    
    print("\n✓ Module testé avec succès!")

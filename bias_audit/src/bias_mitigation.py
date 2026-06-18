"""
Module de techniques d'atténuation des biais.
Implémente les méthodes de pré-traitement, in-processing et post-processing.
"""

import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class PreprocessingDebias:
    """Techniques de débiaisage par pré-traitement des données"""
    
    def __init__(self, protected_attribute: str, privileged_value, unprivileged_value):
        self.protected_attribute = protected_attribute
        self.privileged_value = privileged_value
        self.unprivileged_value = unprivileged_value
    
    def reweighting(self, X: pd.DataFrame, y: pd.Series, 
                   sensitive_features: pd.Series) -> np.ndarray:
        """
        Calcule les poids pour compenser les biais.
        
        Formule de reweighing: w(y,s) = P(y)P(s) / P(y,s)
        """
        weights = np.ones(len(X))
        
        # Calculer les probabilités
        p_y = y.value_counts(normalize=True)
        p_s = sensitive_features.value_counts(normalize=True)
        
        for y_val in y.unique():
            for s_val in sensitive_features.unique():
                mask = (y == y_val) & (sensitive_features == s_val)
                
                if mask.sum() == 0:
                    continue
                
                # P(y, s)
                p_y_s = mask.sum() / len(X)
                
                if p_y_s > 0:
                    w = (p_y[y_val] * p_s[s_val]) / p_y_s
                    weights[mask] = w
        
        # Normaliser les poids
        weights = weights / weights.mean()
        
        return weights
    
    def resampling_balance(self, X: pd.DataFrame, y: pd.Series,
                          sensitive_features: pd.Series,
                          strategy: str = 'oversample') -> Tuple:
        """
        Re-échantillonnage pour équilibrer les groupes.
        
        Args:
            strategy: 'oversample', 'undersample', ou 'smote'
        """
        X_resampled = X.copy()
        y_resampled = y.copy()
        sensitive_resampled = sensitive_features.copy()
        
        if strategy == 'oversample':
            # Equilibrage par couples (groupe sensible, label), plus robuste pour
            # les métriques de TPR/FPR qu'un simple équilibrage par groupe.
            cells = []
            for sensitive_val in sensitive_features.unique():
                for label_val in y.unique():
                    mask = (sensitive_features == sensitive_val) & (y == label_val)
                    if mask.sum() > 0:
                        cells.append((X[mask], y[mask], sensitive_features[mask]))

            target_size = max(len(cell[1]) for cell in cells)
            resampled_cells = []
            for X_cell, y_cell, s_cell in cells:
                replace = len(y_cell) < target_size
                X_part, y_part, s_part = resample(
                    X_cell, y_cell, s_cell,
                    n_samples=target_size,
                    replace=replace,
                    random_state=42
                )
                resampled_cells.append((X_part, y_part, s_part))

            X_resampled = pd.concat([cell[0] for cell in resampled_cells])
            y_resampled = pd.concat([cell[1] for cell in resampled_cells])
            sensitive_resampled = pd.concat([cell[2] for cell in resampled_cells])
        
        elif strategy == 'undersample':
            group_counts = sensitive_features.value_counts()
            majority_group = group_counts.idxmax()
            # Undersample le groupe majoritaire
            majority_mask = sensitive_features == majority_group
            
            X_minority = X[~majority_mask]
            y_minority = y[~majority_mask]
            sensitive_minority = sensitive_features[~majority_mask]
            
            X_majority = X[majority_mask]
            y_majority = y[majority_mask]
            sensitive_majority = sensitive_features[majority_mask]
            
            # Downsample
            X_majority_downsampled, y_majority_downsampled, sensitive_majority_downsampled = resample(
                X_majority, y_majority, sensitive_majority,
                n_samples=len(X_minority),
                random_state=42
            )
            
            # Combiner
            X_resampled = pd.concat([X_minority, X_majority_downsampled])
            y_resampled = pd.concat([y_minority, y_majority_downsampled])
            sensitive_resampled = pd.concat([sensitive_minority, sensitive_majority_downsampled])
        
        elif strategy == 'smote':
            try:
                from imblearn.over_sampling import SMOTE
            except Exception as exc:
                print(
                    "SMOTE indisponible avec cette pile Python 3.14/scikit-learn; "
                    "retour aux données originales."
                )
                return X, y, sensitive_features

            # SMOTE pour équilibrer
            # Préparer les données
            X_with_sensitive = X.copy()
            X_with_sensitive[self.protected_attribute] = sensitive_features
            
            # Créer une colonne combinée pour stratification
            y_combined = y.astype(str) + '_' + sensitive_features.astype(str)
            
            # Appliquer SMOTE
            smote = SMOTE(random_state=42, k_neighbors=3)
            try:
                X_resampled, y_combined_resampled = smote.fit_resample(
                    X_with_sensitive, y_combined
                )
                
                # Séparer les colonnes
                sensitive_resampled = X_resampled[self.protected_attribute]
                X_resampled = X_resampled.drop(columns=[self.protected_attribute])
                
                # Reconstruire y
                y_resampled = pd.Series([int(val.split('_')[0]) for val in y_combined_resampled])
            except:
                print("SMOTE failed, returning original data")
                return X, y, sensitive_features
        
        return X_resampled, y_resampled, sensitive_resampled
    
    def stratified_resampling(self, X: pd.DataFrame, y: pd.Series,
                            sensitive_features: pd.Series) -> Tuple:
        """Re-échantillonnage stratifié par groupe et label"""
        
        # Créer des sous-groupes
        subgroups = []
        for sensitive_val in sensitive_features.unique():
            for label_val in y.unique():
                mask = (sensitive_features == sensitive_val) & (y == label_val)
                subgroups.append({
                    'X': X[mask],
                    'y': y[mask],
                    's': sensitive_features[mask],
                    'size': mask.sum()
                })
        
        # Trouver la taille cible (médiane)
        target_size = int(np.median([sg['size'] for sg in subgroups if sg['size'] > 0]))
        
        # Re-échantillonner chaque sous-groupe
        resampled_data = []
        for sg in subgroups:
            if sg['size'] == 0:
                continue
            
            if sg['size'] < target_size:
                # Oversample
                X_res, y_res, s_res = resample(
                    sg['X'], sg['y'], sg['s'],
                    n_samples=target_size,
                    random_state=42
                )
            elif sg['size'] > target_size:
                # Undersample
                X_res, y_res, s_res = resample(
                    sg['X'], sg['y'], sg['s'],
                    n_samples=target_size,
                    random_state=42,
                    replace=False
                )
            else:
                X_res, y_res, s_res = sg['X'], sg['y'], sg['s']
            
            resampled_data.append((X_res, y_res, s_res))
        
        # Combiner tous les sous-groupes
        X_final = pd.concat([data[0] for data in resampled_data])
        y_final = pd.concat([data[1] for data in resampled_data])
        s_final = pd.concat([data[2] for data in resampled_data])
        
        return X_final, y_final, s_final


class AdversarialDebiasing:
    """
    Adversarial Debiasing - apprentissage adversarial pour créer 
    des représentations aveugles aux attributs protégés.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32],
                 lambda_fairness: float = 1.0):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lambda_fairness = lambda_fairness
        self.model = None
        self.history = None
    
    def build_model(self):
        """Construit le modèle adversarial"""
        try:
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError as exc:
            raise ImportError(
                "AdversarialDebiasing nécessite TensorFlow. "
                "TensorFlow n'est pas inclus dans requirements.txt car il "
                "n'a pas de wheel compatible Python 3.14 dans cette configuration. "
                "Utilisez les méthodes reweighting/resampling/post-processing, "
                "ou un venv Python 3.11 pour TensorFlow."
            ) from exc
        
        # Input
        inputs = keras.Input(shape=(self.input_dim,))
        
        # Shared layers (encoder)
        x = inputs
        for dim in self.hidden_dims:
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.Dropout(0.3)(x)
        
        # Predictor head (tâche principale)
        predictor_hidden = layers.Dense(32, activation='relu', name='predictor_hidden')(x)
        predictor_output = layers.Dense(1, activation='sigmoid', name='predictor_output')(predictor_hidden)
        
        # Adversary head (prédire l'attribut protégé)
        adversary_hidden = layers.Dense(32, activation='relu', name='adversary_hidden')(x)
        
        # Gradient reversal layer (simulé avec Lambda layer)
        def gradient_reversal(x):
            return x  # Le vrai gradient reversal nécessite une implémentation custom
        
        adversary_reversed = layers.Lambda(gradient_reversal, name='gradient_reversal')(adversary_hidden)
        adversary_output = layers.Dense(1, activation='sigmoid', name='adversary_output')(adversary_reversed)
        
        # Modèle complet
        self.model = keras.Model(
            inputs=inputs,
            outputs=[predictor_output, adversary_output]
        )
        
        return self.model
    
    def compile_model(self):
        """Compile le modèle avec les pertes appropriées"""
        try:
            from tensorflow import keras
        except ImportError as exc:
            raise ImportError(
                "AdversarialDebiasing nécessite TensorFlow. "
                "TensorFlow n'est pas inclus dans requirements.txt pour Python 3.14."
            ) from exc
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'predictor_output': 'binary_crossentropy',
                'adversary_output': 'binary_crossentropy'
            },
            loss_weights={
                'predictor_output': 1.0,
                'adversary_output': -self.lambda_fairness  # Négatif pour adversarial
            },
            metrics={
                'predictor_output': ['accuracy'],
                'adversary_output': ['accuracy']
            }
        )
    
    def fit(self, X_train, y_train, sensitive_train,
            X_val=None, y_val=None, sensitive_val=None,
            epochs=50, batch_size=64, verbose=1):
        """Entraîne le modèle adversarial"""
        
        if self.model is None:
            self.build_model()
            self.compile_model()
        
        # Préparer les données de validation
        validation_data = None
        if X_val is not None:
            validation_data = (
                X_val,
                {'predictor_output': y_val, 'adversary_output': sensitive_val}
            )
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_predictor_output_loss' if X_val is not None else 'predictor_output_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_predictor_output_loss' if X_val is not None else 'predictor_output_loss',
                factor=0.5,
                patience=5
            )
        ]
        
        # Entraînement
        self.history = self.model.fit(
            X_train,
            {'predictor_output': y_train, 'adversary_output': sensitive_train},
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return self.history
    
    def predict(self, X):
        """Fait des prédictions (uniquement la sortie principale)"""
        predictor_output, _ = self.model.predict(X, verbose=0)
        return predictor_output.flatten()
    
    def predict_proba(self, X):
        """Retourne les probabilités de prédiction"""
        return self.predict(X)


class PostprocessingDebias:
    """Techniques de débiaisage par post-traitement"""
    
    def __init__(self, protected_attribute: str, privileged_value, unprivileged_value):
        self.protected_attribute = protected_attribute
        self.privileged_value = privileged_value
        self.unprivileged_value = unprivileged_value
        self.thresholds = {}
    
    def calibrated_equalized_odds(self, y_true, y_scores, sensitive_features,
                                  constraint='equalized_odds'):
        """
        Optimise les seuils pour satisfaire les contraintes de fairness.
        
        Args:
            constraint: 'equalized_odds' ou 'demographic_parity'
        """
        from sklearn.metrics import roc_curve
        
        # Calculer les courbes ROC pour chaque groupe
        groups = sensitive_features.unique()
        group_thresholds = {}
        
        for group in groups:
            mask = sensitive_features == group
            
            fpr, tpr, thresholds = roc_curve(y_true[mask], y_scores[mask])
            
            if constraint == 'equalized_odds':
                # Choisir le seuil qui maximise (TPR - FPR)
                optimal_idx = np.argmax(tpr - fpr)
                group_thresholds[group] = thresholds[optimal_idx]
            
            elif constraint == 'demographic_parity':
                # Choisir le seuil qui donne un taux de sélection cible
                target_rate = np.mean(y_scores >= 0.5)  # Taux global
                closest_idx = np.argmin(np.abs(tpr - target_rate))
                group_thresholds[group] = thresholds[closest_idx]
        
        self.thresholds = group_thresholds
        
        # Appliquer les seuils
        y_pred_calibrated = np.zeros_like(y_scores)
        for group in groups:
            mask = sensitive_features == group
            y_pred_calibrated[mask] = (y_scores[mask] >= group_thresholds[group]).astype(int)
        
        return y_pred_calibrated
    
    def reject_option_classification(self, y_scores, sensitive_features,
                                    critical_region_width=0.05,
                                    favorable_to_unprivileged=True):
        """
        Reject Option Based Classification.
        
        Dans une région critique autour du seuil, favorise le groupe défavorisé.
        """
        y_pred = np.zeros_like(y_scores)
        
        # Définir la région critique
        lower_bound = 0.5 - critical_region_width
        upper_bound = 0.5 + critical_region_width
        
        for i, (score, group) in enumerate(zip(y_scores, sensitive_features)):
            if score < lower_bound:
                # En dessous de la région critique -> 0
                y_pred[i] = 0
            elif score > upper_bound:
                # Au-dessus de la région critique -> 1
                y_pred[i] = 1
            else:
                # Dans la région critique -> favoriser groupe défavorisé
                if favorable_to_unprivileged:
                    if group == self.unprivileged_value:
                        y_pred[i] = 1  # Favoriser
                    else:
                        y_pred[i] = 0  # Défavoriser
                else:
                    y_pred[i] = 1 if score >= 0.5 else 0
        
        return y_pred
    
    def platt_scaling_by_group(self, y_true, y_scores, sensitive_features):
        """
        Calibration de Platt (scaling logistique) par groupe.
        """
        from sklearn.linear_model import LogisticRegression
        
        calibrated_scores = np.zeros_like(y_scores)
        
        for group in sensitive_features.unique():
            mask = sensitive_features == group
            
            # Entraîner un modèle logistique pour calibrer
            calibrator = LogisticRegression()
            calibrator.fit(y_scores[mask].reshape(-1, 1), y_true[mask])
            
            # Appliquer la calibration
            calibrated_scores[mask] = calibrator.predict_proba(
                y_scores[mask].reshape(-1, 1)
            )[:, 1]
        
        return calibrated_scores


class FairnessConstrainedModel:
    """Modèle avec contraintes de fairness intégrées"""
    
    def __init__(self, base_model=None, constraint_weight=1.0):
        self.base_model = base_model if base_model else LogisticRegression()
        self.constraint_weight = constraint_weight
    
    def fit_with_fairness_penalty(self, X, y, sensitive_features,
                                  fairness_metric='demographic_parity'):
        """
        Entraîne avec une pénalité de fairness dans la fonction de perte.
        
        Note: Implémentation simplifiée pour LogisticRegression
        """
        # Pour une vraie implémentation, il faudrait une loss custom
        # Ici, on utilise le reweighting comme approximation
        
        from src.data_processing import DataProcessor
        
        # Calculer les poids
        processor = DataProcessor(
            protected_attributes=[sensitive_features.name],
            label_name='target'
        )
        
        # Créer un DataFrame temporaire
        df_temp = X.copy()
        df_temp['target'] = y
        df_temp['sensitive'] = sensitive_features
        
        # Réutiliser la logique de reweighting
        weights = np.ones(len(X))
        
        for y_val in y.unique():
            for s_val in sensitive_features.unique():
                mask = (y == y_val) & (sensitive_features == s_val)
                if mask.sum() > 0:
                    weights[mask] = 1.0 / (mask.sum() / len(X))
        
        # Normaliser
        weights = weights / weights.mean()
        
        # Entraîner avec poids
        self.base_model.fit(X, y, sample_weight=weights)
        
        return self
    
    def predict(self, X):
        return self.base_model.predict(X)
    
    def predict_proba(self, X):
        return self.base_model.predict_proba(X)


def compare_debiasing_methods(X_train, y_train, sensitive_train,
                              X_test, y_test, sensitive_test,
                              methods=['reweighting', 'resampling']):
    """
    Compare différentes méthodes de débiaisage.
    
    Returns:
        Dict avec les résultats de chaque méthode
    """
    from src.metrics import FairnessMetrics, PerformanceMetrics
    from sklearn.linear_model import LogisticRegression
    
    results = {}
    
    # Baseline sans débiaisage
    baseline_model = LogisticRegression(max_iter=1000, random_state=42)
    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)
    y_scores_baseline = baseline_model.predict_proba(X_test)[:, 1]
    
    fm = FairnessMetrics(
        protected_attribute='sensitive',
        privileged_value=sensitive_train.unique()[0],
        unprivileged_value=sensitive_train.unique()[1] if len(sensitive_train.unique()) > 1 else None
    )
    
    results['baseline'] = {
        'fairness': fm.compute_all_metrics(y_test, y_pred_baseline, sensitive_test, y_scores_baseline),
        'performance': PerformanceMetrics.compute_metrics(y_test, y_pred_baseline, y_scores_baseline)
    }
    
    # Méthodes de débiaisage
    if 'reweighting' in methods:
        debias = PreprocessingDebias('sensitive', 
                                     sensitive_train.unique()[0],
                                     sensitive_train.unique()[1] if len(sensitive_train.unique()) > 1 else None)
        weights = debias.reweighting(X_train, y_train, sensitive_train)
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train, sample_weight=weights)
        
        y_pred = model.predict(X_test)
        y_scores = model.predict_proba(X_test)[:, 1]
        
        results['reweighting'] = {
            'fairness': fm.compute_all_metrics(y_test, y_pred, sensitive_test, y_scores),
            'performance': PerformanceMetrics.compute_metrics(y_test, y_pred, y_scores)
        }
    
    if 'resampling' in methods:
        debias = PreprocessingDebias('sensitive',
                                     sensitive_train.unique()[0],
                                     sensitive_train.unique()[1] if len(sensitive_train.unique()) > 1 else None)
        X_res, y_res, s_res = debias.resampling_balance(
            X_train, y_train, sensitive_train, strategy='oversample'
        )
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_res, y_res)
        
        y_pred = model.predict(X_test)
        y_scores = model.predict_proba(X_test)[:, 1]
        
        results['resampling'] = {
            'fairness': fm.compute_all_metrics(y_test, y_pred, sensitive_test, y_scores),
            'performance': PerformanceMetrics.compute_metrics(y_test, y_pred, y_scores)
        }
    
    return results


if __name__ == "__main__":
    print("Test du module de débiaisage...")
    
    # Générer des données synthétiques
    from src.data_processing import generate_sample_data, DataProcessor
    
    df = generate_sample_data(n_samples=2000)
    
    processor = DataProcessor(
        protected_attributes=['gender'],
        label_name='label'
    )
    
    # Encoder
    df_encoded = processor.encode_categorical(df)
    
    # Split
    X_train, X_test, y_train, y_test = processor.split_data(df_encoded)
    
    sensitive_train = X_train['gender']
    sensitive_test = X_test['gender']
    
    X_train_clean = X_train.drop(columns=['gender', 'race'])
    X_test_clean = X_test.drop(columns=['gender', 'race'])
    
    # Test Reweighting
    print("\n=== Test Reweighting ===")
    debias_pre = PreprocessingDebias('gender', 0, 1)
    weights = debias_pre.reweighting(X_train_clean, y_train, sensitive_train)
    print(f"Poids calculés: min={weights.min():.2f}, max={weights.max():.2f}, mean={weights.mean():.2f}")
    
    # Test Resampling
    print("\n=== Test Resampling ===")
    X_res, y_res, s_res = debias_pre.resampling_balance(
        X_train_clean, y_train, sensitive_train, strategy='oversample'
    )
    print(f"Taille originale: {len(X_train_clean)}")
    print(f"Taille après resampling: {len(X_res)}")
    print(f"Distribution après resampling: {s_res.value_counts()}")
    
    # Test Adversarial Debiasing
    print("\n=== Test Adversarial Debiasing ===")
    print("Ignoré: TensorFlow n'est pas inclus dans l'environnement Python 3.14 par défaut.")
    
    print("\n✓ Module de débiaisage testé avec succès!")

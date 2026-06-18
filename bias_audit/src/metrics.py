"""
Module de calcul des métriques de fairness et de performance.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from typing import Callable, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def _as_array(values) -> np.ndarray:
    """Convertit pandas/listes en tableau numpy plat."""
    return np.asarray(values).reshape(-1)


def _safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    return float(numerator / denominator) if denominator else default


def binary_confusion_rates(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Retourne les taux de confusion principaux pour une classification binaire."""
    cm = confusion_matrix(_as_array(y_true), _as_array(y_pred), labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'tpr': _safe_divide(tp, tp + fn),
        'fpr': _safe_divide(fp, fp + tn),
        'tnr': _safe_divide(tn, tn + fp),
        'fnr': _safe_divide(fn, fn + tp),
    }


def compute_group_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                          sensitive_features: np.ndarray,
                          y_scores: np.ndarray = None) -> Dict[str, Dict]:
    """Calcule performance, selection rate, TPR/FPR et base rate pour chaque groupe."""
    y_true = _as_array(y_true)
    y_pred = _as_array(y_pred)
    sensitive_features = _as_array(sensitive_features)
    y_scores = _as_array(y_scores) if y_scores is not None else None

    groups = pd.Series(sensitive_features).dropna().unique()
    results = {}
    for group in groups:
        mask = sensitive_features == group
        if mask.sum() == 0:
            continue
        group_scores = y_scores[mask] if y_scores is not None else None
        perf = PerformanceMetrics.compute_metrics(y_true[mask], y_pred[mask], group_scores)
        rates = binary_confusion_rates(y_true[mask], y_pred[mask])
        results[str(group)] = {
            'sample_size': int(mask.sum()),
            'base_rate': float(np.mean(y_true[mask])),
            'selection_rate': float(np.mean(y_pred[mask])),
            'accuracy': float(perf['accuracy']),
            'precision': float(perf['precision']),
            'recall': float(perf['recall']),
            'f1_score': float(perf['f1_score']),
            'tpr': float(rates['tpr']),
            'fpr': float(rates['fpr']),
            'tnr': float(rates['tnr']),
            'fnr': float(rates['fnr']),
        }
        if 'roc_auc' in perf and perf['roc_auc'] is not None:
            results[str(group)]['roc_auc'] = float(perf['roc_auc'])
    return results


def compute_multigroup_fairness(y_true: np.ndarray, y_pred: np.ndarray,
                                sensitive_features: np.ndarray,
                                y_scores: np.ndarray = None) -> Dict:
    """
    Calcule des métriques de fairness valables pour deux groupes ou plus.

    Equalized odds difference est le maximum entre l'écart de TPR et l'écart de FPR,
    conformément à la définition usuelle utilisée dans les audits.
    """
    group_metrics = compute_group_metrics(y_true, y_pred, sensitive_features, y_scores)
    if not group_metrics:
        return {
            'group_metrics': {},
            'demographic_parity_difference': 0.0,
            'demographic_parity_ratio': 1.0,
            'disparate_impact': 1.0,
            'tpr_difference': 0.0,
            'fpr_difference': 0.0,
            'equalized_odds_difference': 0.0,
            'average_odds_difference': 0.0,
            'equal_opportunity_difference': 0.0,
            'predictive_parity_difference': 0.0,
        }

    selection_rates = np.array([m['selection_rate'] for m in group_metrics.values()])
    tprs = np.array([m['tpr'] for m in group_metrics.values()])
    fprs = np.array([m['fpr'] for m in group_metrics.values()])
    precisions = np.array([m['precision'] for m in group_metrics.values()])

    min_selection = float(selection_rates.min())
    max_selection = float(selection_rates.max())
    dp_diff = max_selection - min_selection
    dp_ratio = _safe_divide(min_selection, max_selection, default=1.0)
    tpr_diff = float(tprs.max() - tprs.min())
    fpr_diff = float(fprs.max() - fprs.min())

    return {
        'group_metrics': group_metrics,
        'demographic_parity_difference': float(dp_diff),
        'demographic_parity_ratio': float(dp_ratio),
        'disparate_impact': float(dp_ratio),
        'tpr_difference': tpr_diff,
        'fpr_difference': fpr_diff,
        'equalized_odds_difference': float(max(tpr_diff, fpr_diff)),
        'average_odds_difference': float(0.5 * (tpr_diff + fpr_diff)),
        'equal_opportunity_difference': tpr_diff,
        'predictive_parity_difference': float(precisions.max() - precisions.min()),
    }


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_iterations: int = 200,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> Dict[str, float]:
    """Intervalle de confiance bootstrap pour une métrique binaire."""
    y_true = _as_array(y_true)
    y_pred = _as_array(y_pred)
    rng = np.random.default_rng(random_state)
    values = []
    n = len(y_true)

    if n == 0:
        return {'mean': 0.0, 'lower': 0.0, 'upper': 0.0}

    for _ in range(n_iterations):
        idx = rng.integers(0, n, n)
        try:
            values.append(float(metric_fn(y_true[idx], y_pred[idx])))
        except Exception:
            continue

    if not values:
        return {'mean': 0.0, 'lower': 0.0, 'upper': 0.0}

    alpha = 1.0 - confidence_level
    return {
        'mean': float(np.mean(values)),
        'lower': float(np.quantile(values, alpha / 2)),
        'upper': float(np.quantile(values, 1 - alpha / 2)),
    }


def summarize_numeric_rows(rows: List[Dict], group_keys: List[str]) -> List[Dict]:
    """Moyenne et écart-type par groupe de colonnes pour une liste de résultats."""
    if not rows:
        return []
    df = pd.DataFrame(rows)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    summaries = []
    for keys, group in df.groupby(group_keys, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {key: value for key, value in zip(group_keys, keys)}
        for col in numeric_cols:
            row[f'{col}_mean'] = float(group[col].mean())
            row[f'{col}_std'] = float(group[col].std(ddof=0))
        summaries.append(row)
    return summaries


class FairnessMetrics:
    """Calculateur de métriques de fairness"""
    
    def __init__(self, protected_attribute: str, privileged_value, unprivileged_value):
        """
        Args:
            protected_attribute: Nom de l'attribut protégé
            privileged_value: Valeur du groupe privilégié
            unprivileged_value: Valeur du groupe défavorisé
        """
        self.protected_attribute = protected_attribute
        self.privileged_value = privileged_value
        self.unprivileged_value = unprivileged_value
    
    def demographic_parity_difference(self, y_pred: np.ndarray, 
                                     sensitive_features: np.ndarray) -> float:
        """
        Calcule la différence de parité démographique.
        
        DPD = P(Ŷ=1|A=privileged) - P(Ŷ=1|A=unprivileged)
        Idéalement proche de 0. Valeurs acceptables: [-0.1, 0.1]
        """
        y_pred = _as_array(y_pred)
        sensitive_features = _as_array(sensitive_features)
        priv_mask = sensitive_features == self.privileged_value
        unpriv_mask = sensitive_features == self.unprivileged_value
        if priv_mask.sum() == 0 or unpriv_mask.sum() == 0:
            return 0.0
        
        priv_rate = np.mean(y_pred[priv_mask])
        unpriv_rate = np.mean(y_pred[unpriv_mask])
        
        return priv_rate - unpriv_rate
    
    def demographic_parity_ratio(self, y_pred: np.ndarray,
                                sensitive_features: np.ndarray) -> float:
        """
        Calcule le ratio de parité démographique.
        
        DPR = P(Ŷ=1|A=unprivileged) / P(Ŷ=1|A=privileged)
        Idéalement proche de 1. Valeurs acceptables: [0.8, 1.25]
        """
        y_pred = _as_array(y_pred)
        sensitive_features = _as_array(sensitive_features)
        priv_mask = sensitive_features == self.privileged_value
        unpriv_mask = sensitive_features == self.unprivileged_value
        if priv_mask.sum() == 0 or unpriv_mask.sum() == 0:
            return 1.0
        
        priv_rate = np.mean(y_pred[priv_mask])
        unpriv_rate = np.mean(y_pred[unpriv_mask])
        
        if priv_rate == 0:
            return 1.0 if unpriv_rate == 0 else np.inf
        
        return unpriv_rate / priv_rate
    
    def disparate_impact(self, y_pred: np.ndarray,
                        sensitive_features: np.ndarray) -> float:
        """
        Calcule le disparate impact (ratio des taux de sélection).
        
        DI = P(Ŷ=1|A=unprivileged) / P(Ŷ=1|A=privileged)
        Règle des 80%: DI devrait être >= 0.8
        """
        return self.demographic_parity_ratio(y_pred, sensitive_features)
    
    def equalized_odds_difference(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  sensitive_features: np.ndarray) -> Dict[str, float]:
        """
        Calcule la différence d'equalized odds.
        
        Mesure la différence de TPR et FPR entre groupes.
        EOD_TPR = TPR(privileged) - TPR(unprivileged)
        EOD_FPR = FPR(privileged) - FPR(unprivileged)
        """
        y_true = _as_array(y_true)
        y_pred = _as_array(y_pred)
        sensitive_features = _as_array(sensitive_features)
        priv_mask = sensitive_features == self.privileged_value
        unpriv_mask = sensitive_features == self.unprivileged_value
        if priv_mask.sum() == 0 or unpriv_mask.sum() == 0:
            return {
                'tpr_difference': 0.0,
                'fpr_difference': 0.0,
                'average_odds_difference': 0.0,
                'equalized_odds_difference': 0.0,
            }
        
        # True Positive Rate
        tpr_priv = self._tpr(y_true[priv_mask], y_pred[priv_mask])
        tpr_unpriv = self._tpr(y_true[unpriv_mask], y_pred[unpriv_mask])
        
        # False Positive Rate
        fpr_priv = self._fpr(y_true[priv_mask], y_pred[priv_mask])
        fpr_unpriv = self._fpr(y_true[unpriv_mask], y_pred[unpriv_mask])
        
        tpr_difference = tpr_priv - tpr_unpriv
        fpr_difference = fpr_priv - fpr_unpriv
        return {
            'tpr_difference': tpr_difference,
            'fpr_difference': fpr_difference,
            'average_odds_difference': 0.5 * (tpr_difference + fpr_difference),
            'equalized_odds_difference': max(abs(tpr_difference), abs(fpr_difference)),
        }
    
    def equal_opportunity_difference(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     sensitive_features: np.ndarray) -> float:
        """
        Calcule la différence d'égalité des opportunités.
        
        EOppD = TPR(privileged) - TPR(unprivileged)
        Se concentre uniquement sur les vrais positifs.
        """
        y_true = _as_array(y_true)
        y_pred = _as_array(y_pred)
        sensitive_features = _as_array(sensitive_features)
        priv_mask = sensitive_features == self.privileged_value
        unpriv_mask = sensitive_features == self.unprivileged_value
        if priv_mask.sum() == 0 or unpriv_mask.sum() == 0:
            return 0.0
        
        tpr_priv = self._tpr(y_true[priv_mask], y_pred[priv_mask])
        tpr_unpriv = self._tpr(y_true[unpriv_mask], y_pred[unpriv_mask])
        
        return tpr_priv - tpr_unpriv
    
    def predictive_parity_difference(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     sensitive_features: np.ndarray) -> float:
        """
        Calcule la différence de parité prédictive.
        
        PPD = Precision(privileged) - Precision(unprivileged)
        """
        y_true = _as_array(y_true)
        y_pred = _as_array(y_pred)
        sensitive_features = _as_array(sensitive_features)
        priv_mask = sensitive_features == self.privileged_value
        unpriv_mask = sensitive_features == self.unprivileged_value
        if priv_mask.sum() == 0 or unpriv_mask.sum() == 0:
            return 0.0
        
        # Précision pour chaque groupe
        prec_priv = precision_score(y_true[priv_mask], y_pred[priv_mask], zero_division=0)
        prec_unpriv = precision_score(y_true[unpriv_mask], y_pred[unpriv_mask], zero_division=0)
        
        return prec_priv - prec_unpriv
    
    def statistical_parity_difference(self, y_pred: np.ndarray,
                                     sensitive_features: np.ndarray) -> float:
        """Alias pour demographic_parity_difference"""
        return self.demographic_parity_difference(y_pred, sensitive_features)
    
    def compute_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                           sensitive_features: np.ndarray, y_scores: np.ndarray = None) -> Dict:
        """Calcule toutes les métriques de fairness"""
        
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
        metrics['group_metrics'] = compute_group_metrics(
            y_true, y_pred, sensitive_features, y_scores
        )
        metrics['interpretation'] = self._interpret_metrics(metrics)
        
        return metrics
    
    def _tpr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcule le True Positive Rate"""
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape[0] < 2 or cm.shape[1] < 2 or (cm[1, 0] + cm[1, 1]) == 0:
            return 0.0
        return cm[1, 1] / (cm[1, 0] + cm[1, 1])
    
    def _fpr(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcule le False Positive Rate"""
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape[0] < 2 or cm.shape[1] < 2 or (cm[0, 0] + cm[0, 1]) == 0:
            return 0.0
        return cm[0, 1] / (cm[0, 0] + cm[0, 1])
    
    def _interpret_metrics(self, metrics: Dict) -> Dict[str, str]:
        """Interprète les métriques calculées"""
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
        equalized_odds = abs(metrics.get('equalized_odds_difference', metrics['average_odds_difference']))
        if equalized_odds <= 0.1:
            interpretation['equalized_odds'] = "✓ Bonnes odds égalisées"
        elif equalized_odds <= 0.2:
            interpretation['equalized_odds'] = "⚠ Odds égalisées modérées"
        else:
            interpretation['equalized_odds'] = "✗ Odds égalisées faibles"
        
        return interpretation


class PerformanceMetrics:
    """Calculateur de métriques de performance"""
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                       y_scores: np.ndarray = None) -> Dict:
        """Calcule les métriques de performance classiques"""
        
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
        """Calcule les métriques stratifiées par groupe"""
        
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
    """Compare les métriques entre modèle baseline et débiaisé"""
    
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
    
    print("\nMétriques de Fairness:")
    for metric, value in fairness_metrics.items():
        if metric != 'interpretation':
            print(f"  {metric}: {value}")
    
    print("\nInterprétation:")
    for metric, interp in fairness_metrics['interpretation'].items():
        print(f"  {metric}: {interp}")
    
    # Test PerformanceMetrics
    perf_metrics = PerformanceMetrics.compute_metrics(y_true, y_pred, y_scores)
    
    print("\nMétriques de Performance:")
    for metric, value in perf_metrics.items():
        if metric != 'confusion_matrix':
            print(f"  {metric}: {value}")
    
    # Test métriques stratifiées
    stratified = PerformanceMetrics.compute_stratified_metrics(
        y_true, y_pred, sensitive, ['A', 'B'], y_scores
    )
    
    print("\nMétriques Stratifiées:")
    for group, metrics in stratified.items():
        print(f"  Groupe {group}:")
        print(f"    Accuracy: {metrics['accuracy']:.4f}")
        print(f"    F1-Score: {metrics['f1_score']:.4f}")
    
    print("\n✓ Module testé avec succès!")

# Fichier: tests/test_metrics.py
"""
Tests unitaires pour le module metrics.
"""

import pytest
import numpy as np
import pandas as pd
from src.metrics import FairnessMetrics, PerformanceMetrics


class TestFairnessMetrics:
    """Tests pour FairnessMetrics"""
    
    @pytest.fixture
    def sample_data(self):
        """Crée des données de test"""
        np.random.seed(42)
        n = 1000
        
        y_true = np.random.randint(0, 2, n)
        y_pred = np.random.randint(0, 2, n)
        sensitive = np.random.choice([0, 1], n, p=[0.6, 0.4])
        y_scores = np.random.random(n)
        
        return y_true, y_pred, sensitive, y_scores
    
    def test_demographic_parity_difference(self, sample_data):
        """Test calcul demographic parity difference"""
        y_true, y_pred, sensitive, y_scores = sample_data
        
        fm = FairnessMetrics('test_attr', privileged_value=0, unprivileged_value=1)
        dpd = fm.demographic_parity_difference(y_pred, sensitive)
        
        # Vérifications
        assert isinstance(dpd, (int, float, np.number))
        assert -1 <= dpd <= 1  # DPD doit être entre -1 et 1
    
    def test_disparate_impact(self, sample_data):
        """Test calcul disparate impact"""
        y_true, y_pred, sensitive, y_scores = sample_data
        
        fm = FairnessMetrics('test_attr', privileged_value=0, unprivileged_value=1)
        di = fm.disparate_impact(y_pred, sensitive)
        
        # Vérifications
        assert isinstance(di, (int, float, np.number))
        assert di >= 0  # DI doit être positif
    
    def test_equalized_odds(self, sample_data):
        """Test calcul equalized odds"""
        y_true, y_pred, sensitive, y_scores = sample_data
        
        fm = FairnessMetrics('test_attr', privileged_value=0, unprivileged_value=1)
        eod = fm.equalized_odds_difference(y_true, y_pred, sensitive)
        
        # Vérifications
        assert isinstance(eod, dict)
        assert 'tpr_difference' in eod
        assert 'fpr_difference' in eod
        assert 'average_odds_difference' in eod
        
        # Les différences doivent être entre -1 et 1
        assert -1 <= eod['tpr_difference'] <= 1
        assert -1 <= eod['fpr_difference'] <= 1
    
    def test_compute_all_metrics(self, sample_data):
        """Test calcul de toutes les métriques"""
        y_true, y_pred, sensitive, y_scores = sample_data
        
        fm = FairnessMetrics('test_attr', privileged_value=0, unprivileged_value=1)
        metrics = fm.compute_all_metrics(y_true, y_pred, sensitive, y_scores)
        
        # Vérifications
        assert isinstance(metrics, dict)
        assert 'demographic_parity_difference' in metrics
        assert 'disparate_impact' in metrics
        assert 'average_odds_difference' in metrics
        assert 'interpretation' in metrics
    
    def test_perfect_fairness(self):
        """Test avec fairness parfaite"""
        n = 100
        y_true = np.array([0, 1] * 50)
        y_pred = np.array([0, 1] * 50)
        sensitive = np.array([0, 0, 1, 1] * 25)
        y_scores = np.array([0.2, 0.8] * 50)
        
        fm = FairnessMetrics('test_attr', privileged_value=0, unprivileged_value=1)
        
        dpd = fm.demographic_parity_difference(y_pred, sensitive)
        di = fm.disparate_impact(y_pred, sensitive)
        
        # Avec distribution parfaite, DPD devrait être ~0 et DI ~1
        assert abs(dpd) < 0.01
        assert 0.99 <= di <= 1.01
    
    def test_extreme_bias(self):
        """Test avec biais extrême"""
        n = 100
        y_true = np.ones(n)
        y_pred = np.concatenate([np.ones(50), np.zeros(50)])
        sensitive = np.concatenate([np.zeros(50), np.ones(50)])
        
        fm = FairnessMetrics('test_attr', privileged_value=0, unprivileged_value=1)
        
        di = fm.disparate_impact(y_pred, sensitive)
        
        # Biais extrême: groupe défavorisé a 0% de positifs
        assert di == 0


class TestPerformanceMetrics:
    """Tests pour PerformanceMetrics"""
    
    @pytest.fixture
    def sample_predictions(self):
        """Crée des prédictions de test"""
        np.random.seed(42)
        n = 1000
        
        y_true = np.random.randint(0, 2, n)
        y_pred = np.random.randint(0, 2, n)
        y_scores = np.random.random(n)
        
        return y_true, y_pred, y_scores
    
    def test_compute_metrics(self, sample_predictions):
        """Test calcul métriques de performance"""
        y_true, y_pred, y_scores = sample_predictions
        
        metrics = PerformanceMetrics.compute_metrics(y_true, y_pred, y_scores)
        
        # Vérifications
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        
        # Toutes les métriques doivent être entre 0 et 1
        for key in ['accuracy', 'precision', 'recall', 'f1_score']:
            assert 0 <= metrics[key] <= 1
    
    def test_perfect_predictions(self):
        """Test avec prédictions parfaites"""
        n = 100
        y_true = np.array([0, 1] * 50)
        y_pred = np.array([0, 1] * 50)
        y_scores = np.array([0.1, 0.9] * 50)
        
        metrics = PerformanceMetrics.compute_metrics(y_true, y_pred, y_scores)
        
        # Avec prédictions parfaites, toutes les métriques = 1
        assert metrics['accuracy'] == 1.0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 1.0
        assert metrics['f1_score'] == 1.0
    
    def test_stratified_metrics(self):
        """Test métriques stratifiées"""
        np.random.seed(42)
        n = 1000
        
        y_true = np.random.randint(0, 2, n)
        y_pred = np.random.randint(0, 2, n)
        sensitive = np.random.choice(['A', 'B'], n)
        y_scores = np.random.random(n)
        
        stratified = PerformanceMetrics.compute_stratified_metrics(
            y_true, y_pred, sensitive, ['A', 'B'], y_scores
        )
        
        # Vérifications
        assert isinstance(stratified, dict)
        assert 'A' in stratified
        assert 'B' in stratified
        assert 'accuracy' in stratified['A']
        assert 'sample_size' in stratified['A']

# Fichier: tests/conftest.py
"""
Configuration pytest et fixtures partagées.
"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture(scope="session")
def random_seed():
    """Seed aléatoire fixe pour tous les tests"""
    return 42


@pytest.fixture(scope="session")
def test_data_size():
    """Taille standard des données de test"""
    return 1000


@pytest.fixture
def simple_binary_data():
    """Données binaires simples pour tests rapides"""
    np.random.seed(42)
    n = 100
    
    return {
        'y_true': np.random.randint(0, 2, n),
        'y_pred': np.random.randint(0, 2, n),
        'sensitive': np.random.choice([0, 1], n),
        'y_scores': np.random.random(n)
    }
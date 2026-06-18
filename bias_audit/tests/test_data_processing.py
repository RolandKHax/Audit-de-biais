# Fichier: tests/test_data_processing.py
"""
Tests unitaires pour le module data_processing.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_processing import DataProcessor, generate_sample_data, prepare_compas_data


class TestDataProcessor:
    """Tests pour DataProcessor"""
    
    @pytest.fixture
    def sample_df(self):
        """Crée un DataFrame de test"""
        return generate_sample_data(n_samples=1000, random_state=42)
    
    @pytest.fixture
    def processor(self):
        """Crée un DataProcessor"""
        return DataProcessor(
            protected_attributes=['gender', 'race'],
            label_name='label'
        )
    
    def test_explore_demographics(self, processor, sample_df):
        """Test analyse démographique"""
        demographics = processor.explore_demographics(sample_df)
        
        # Vérifications
        assert isinstance(demographics, dict)
        assert 'gender' in demographics
        assert 'race' in demographics
        assert 'label' in demographics
        
        # Chaque attribut doit avoir counts et proportions
        for attr in ['gender', 'race']:
            assert 'counts' in demographics[attr]
            assert 'proportions' in demographics[attr]
            assert 'unique_values' in demographics[attr]
    
    def test_check_data_quality(self, processor, sample_df):
        """Test vérification qualité"""
        quality = processor.check_data_quality(sample_df)
        
        # Vérifications
        assert isinstance(quality, dict)
        assert 'missing_values' in quality
        assert 'duplicates' in quality
        assert 'outliers' in quality
        assert 'dtypes' in quality
    
    def test_identify_proxies(self, processor, sample_df):
        """Test identification des proxies"""
        proxies = processor.identify_proxies(sample_df, threshold=0.3)
        
        # Vérifications
        assert isinstance(proxies, dict)
        assert 'gender' in proxies
        assert 'race' in proxies
        
        # Les proxies doivent être une liste de dictionnaires
        for attr, proxy_list in proxies.items():
            assert isinstance(proxy_list, list)
            if len(proxy_list) > 0:
                assert 'feature' in proxy_list[0]
                assert 'correlation' in proxy_list[0]
                assert 'p_value' in proxy_list[0]
    
    def test_encode_categorical(self, processor, sample_df):
        """Test encodage catégoriel"""
        df_encoded = processor.encode_categorical(sample_df.copy())
        
        # Vérifications
        assert df_encoded.shape == sample_df.shape
        
        # Les colonnes catégorielles doivent être encodées en numérique
        for col in ['gender', 'race', 'education']:
            assert df_encoded[col].dtype in [np.int64, np.int32]
    
    def test_split_data(self, processor, sample_df):
        """Test split des données"""
        df_encoded = processor.encode_categorical(sample_df.copy())
        
        X_train, X_test, y_train, y_test = processor.split_data(
            df_encoded, test_size=0.2, random_state=42
        )
        
        # Vérifications
        total_size = len(sample_df)
        assert len(X_train) == int(total_size * 0.8)
        assert len(X_test) == total_size - len(X_train)
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)
        
        # Vérifier que le split est stratifié
        train_dist = y_train.value_counts(normalize=True)
        test_dist = y_test.value_counts(normalize=True)
        
        # Les distributions doivent être similaires (tolérance 5%)
        for label in train_dist.index:
            assert abs(train_dist[label] - test_dist[label]) < 0.05


class TestGenerateSampleData:
    """Tests pour generate_sample_data"""
    
    def test_sample_data_shape(self):
        """Test forme du dataset généré"""
        df = generate_sample_data(n_samples=5000, random_state=42)
        
        assert df.shape[0] == 5000
        assert df.shape[1] == 7  # 6 features + 1 label
    
    def test_sample_data_columns(self):
        """Test colonnes du dataset"""
        df = generate_sample_data(n_samples=1000, random_state=42)
        
        expected_cols = ['gender', 'race', 'age', 'education', 
                        'years_experience', 'test_score', 'label']
        
        assert list(df.columns) == expected_cols
    
    def test_sample_data_types(self):
        """Test types des colonnes"""
        df = generate_sample_data(n_samples=1000, random_state=42)
        
        # Colonnes catégorielles
        assert df['gender'].dtype == 'object'
        assert df['race'].dtype == 'object'
        assert df['education'].dtype == 'object'
        
        # Colonnes numériques
        assert df['age'].dtype in [np.int64, np.int32]
        assert df['years_experience'].dtype in [np.int64, np.int32]
        assert df['test_score'].dtype in [np.float64, np.float32]
        assert df['label'].dtype in [np.int64, np.int32]
    
    def test_sample_data_bias(self):
        """Test présence de biais intentionnel"""
        df = generate_sample_data(n_samples=10000, random_state=42)
        
        # Vérifier que le biais est présent
        male_rate = df[df['gender'] == 'Male']['label'].mean()
        female_rate = df[df['gender'] == 'Female']['label'].mean()
        
        # Les hommes devraient avoir un taux de positifs plus élevé
        assert male_rate > female_rate
        
        # Le ratio devrait violer la règle des 80%
        assert female_rate / male_rate < 0.8
    
    def test_reproducibility(self):
        """Test reproductibilité avec random_state"""
        df1 = generate_sample_data(n_samples=1000, random_state=42)
        df2 = generate_sample_data(n_samples=1000, random_state=42)
        
        # Les deux datasets doivent être identiques
        pd.testing.assert_frame_equal(df1, df2)


class TestCompasPreprocessing:
    """Tests du preset COMPAS."""

    def test_prepare_compas_data_filters_and_keeps_core_columns(self):
        raw = pd.DataFrame({
            "age": [25, 30, 40],
            "age_cat": ["Less than 25", "25 - 45", "25 - 45"],
            "race": ["African-American", "Caucasian", "Hispanic"],
            "sex": ["Male", "Female", "Male"],
            "priors_count": [0, 2, 1],
            "c_charge_degree": ["F", "M", "O"],
            "juv_fel_count": [0, 0, 0],
            "juv_misd_count": [0, 1, 0],
            "juv_other_count": [0, 0, 0],
            "decile_score": [3, 7, 5],
            "score_text": ["Low", "High", "Medium"],
            "days_b_screening_arrest": [0, 5, 2],
            "is_recid": [0, 1, 0],
            "two_year_recid": [0, 1, 0],
        })

        compas = prepare_compas_data(raw)

        assert list(compas["c_charge_degree"]) == ["F", "M"]
        assert "race" in compas.columns
        assert "sex" in compas.columns
        assert "two_year_recid" in compas.columns
        assert compas["two_year_recid"].dtype in [np.int64, np.int32]

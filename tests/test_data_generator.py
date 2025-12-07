import pytest
import numpy as np
from src.data.data_generator import DataGenerator

@pytest.fixture
def base_config():
    return {
        'random_state': 42,
        'dataset': {
            'n_samples': 1000,
            'n_features': 10,
            'n_informative': 5,
            'n_redundant': 2,
            'n_repeated': 0,
            'n_classes': 2,
            'weights': [0.7, 0.3],
            'noise': 0.05,
            'flip_y': 0.01
        },
        'missingness': {
            'mechanism': 'mcar',
            'missing_rate': 0.2
        }
    }

def test_oracle_generation(base_config):
    gen = DataGenerator(base_config)
    data = gen.generate_oracle_data()
    
    assert data['X'].shape == (1000, 10)
    assert data['y'].shape == (1000,)
    assert len(np.unique(data['y'])) <= 2

def test_mcar_mechanism(base_config):
    base_config['missingness']['mechanism'] = 'mcar'
    base_config['missingness']['missing_rate'] = 0.3
    
    gen = DataGenerator(base_config)
    data = gen.generate_oracle_data()
    data_miss = gen.introduce_missingness(data)
    
    mask = data_miss['mask']
    observed_rate = mask.mean()
    
    # Check if observed rate is close to 1 - missing_rate (0.7)
    # Allow small tolerance due to randomness
    assert 0.65 < observed_rate < 0.75
    
    # Check y_observed has NaNs where mask is 0
    assert np.isnan(data_miss['y_observed'][mask == 0]).all()
    assert not np.isnan(data_miss['y_observed'][mask == 1]).any()

def test_mnar_rejection_mechanism(base_config):
    base_config['missingness']['mechanism'] = 'mnar_rejection'
    base_config['missingness']['missing_rate'] = 0.4
    
    gen = DataGenerator(base_config)
    data = gen.generate_oracle_data()
    data_miss = gen.introduce_missingness(data)
    
    mask = data_miss['mask']
    observed_rate = mask.mean()
    
    assert 0.55 < observed_rate < 0.65

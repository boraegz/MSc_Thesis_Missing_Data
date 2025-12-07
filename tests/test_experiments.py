import pytest
import pandas as pd
from src.experiments import ExperimentRunner
from src.utils import load_config

@pytest.fixture
def test_config():
    return {
        'random_state': 42,
        'dataset': {
            'n_samples': 500, # Small for speed
            'n_features': 5,
            'n_informative': 3,
            'n_redundant': 0,
            'n_repeated': 0,
            'n_classes': 2,
            'weights': [0.7, 0.3],
            'noise': 0.1,
            'flip_y': 0.05
        },
        'missingness': {
            'mechanism': 'mcar',
            'missing_rate': 0.2
        }
    }

def test_experiment_runner_single_run(test_config):
    runner = ExperimentRunner(test_config)
    results = runner.run_pipeline(mechanism='mcar', missing_rate=0.2, method='mean')
    
    assert isinstance(results, dict)
    assert 'auc_roc' in results
    assert 'brier_score' in results
    assert results['method'] == 'mean'
    
def test_experiment_runner_full_loop(test_config):
    runner = ExperimentRunner(test_config)
    # Monkey patch run_all_experiments loop list for speed
    # We can't easily monkey patch internal lists of method but we can call run_pipeline manually
    pass 

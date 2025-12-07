import pytest
import numpy as np
from src.imputation.baseline_imputer import BaselineImputer
from src.imputation.mice_imputer import MiceImputer
from src.imputation.missforest_imputer import MissForestImputer
from src.reject_inference.reweighting import InverseProbabilityWeighter
from src.reject_inference.augmentation import Augmentation
from sklearn.linear_model import LogisticRegression

@pytest.fixture
def data_with_nans():
    X = np.array([
        [1.0, 2.0],
        [np.nan, 3.0],
        [4.0, np.nan],
        [7.0, 8.0]
    ])
    return X

@pytest.fixture
def rejection_data():
    X = np.random.rand(100, 5)
    y_observed = np.random.randint(0, 2, 100).astype(float)
    mask = np.random.randint(0, 2, 100) # 0 or 1
    # set rejected y to nan
    y_observed[mask == 0] = np.nan
    return X, y_observed, mask

def test_baseline_imputer(data_with_nans):
    imputer = BaselineImputer(strategy='mean')
    X_imp = imputer.fit(data_with_nans).transform(data_with_nans)
    assert not np.isnan(X_imp).any()
    assert X_imp.shape == data_with_nans.shape
    # Check mean imputation logic
    # First col mean: (1+4+7)/3 = 4.0. NaN should be 4.0
    assert X_imp[1, 0] == 4.0

def test_mice_imputer(data_with_nans):
    imputer = MiceImputer(max_iter=5)
    X_imp = imputer.fit(data_with_nans).transform(data_with_nans)
    assert not np.isnan(X_imp).any()
    assert X_imp.shape == data_with_nans.shape

def test_missforest_imputer(data_with_nans):
    imputer = MissForestImputer(max_iter=2, n_estimators=10)
    X_imp = imputer.fit(data_with_nans).transform(data_with_nans)
    assert not np.isnan(X_imp).any()
    assert X_imp.shape == data_with_nans.shape

def test_reweighting(rejection_data):
    X, y_miss, mask = rejection_data
    # IPW fits on X and mask
    ipw = InverseProbabilityWeighter()
    ipw.fit(X, mask)
    weights = ipw.get_weights(X)
    
    assert len(weights) == len(X)
    assert (weights > 0).all()

def test_augmentation(rejection_data):
    X, y_miss, mask = rejection_data
    aug = Augmentation(base_estimator=LogisticRegression(), soft_labels=False)
    
    # Fit returns self
    aug.fit(X, y_miss, mask)
    
    # Transform returns dataset
    X_aug, y_aug = aug.transform(X, y_miss, mask)
    
    assert X_aug.shape == X.shape
    assert len(y_aug) == len(y_miss)
    assert not np.isnan(y_aug).any() # Standard behavior: fill all
    
    # Check that observed labels are preserved
    assert np.allclose(y_aug[mask == 1], y_miss[mask == 1])

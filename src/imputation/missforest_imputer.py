import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, TransformerMixin

class MissForestImputer(BaseEstimator, TransformerMixin):
    """
    Implements MissForest algorithm using IterativeImputer with a RandomForestRegressor.
    """
    def __init__(self, max_iter: int = 10, n_estimators: int = 100, random_state: int = 42):
        self.max_iter = max_iter
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        # MissForest is essentially IterativeImputer with a RF estimator
        self.imputer = IterativeImputer(
            estimator=RandomForestRegressor(
                n_estimators=n_estimators,
                n_jobs=-1,
                random_state=random_state
            ),
            max_iter=max_iter,
            random_state=random_state
        )
        
    def fit(self, X, y=None):
        self.imputer.fit(X)
        return self
        
    def transform(self, X):
        return self.imputer.transform(X)

import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.base import BaseEstimator, TransformerMixin

class MiceImputer(BaseEstimator, TransformerMixin):
    """
    Implements MICE (Multiple Imputation by Chained Equations) using sklearn's IterativeImputer.
    Default estimator is BayesianRidge.
    """
    def __init__(self, max_iter: int = 10, random_state: int = 42):
        self.max_iter = max_iter
        self.random_state = random_state
        self.imputer = IterativeImputer(
            estimator=BayesianRidge(),
            max_iter=max_iter,
            random_state=random_state,
            sample_posterior=False # Set to True for full MICE variability, but usually False for single imputation
        )
        
    def fit(self, X, y=None):
        self.imputer.fit(X)
        return self

    def transform(self, X):
        return self.imputer.transform(X)

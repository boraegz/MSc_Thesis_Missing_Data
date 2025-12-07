import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

class BaselineImputer(BaseEstimator, TransformerMixin):
    """
    Implements baseline imputation strategies: Mean, Median, Mode, Zero.
    """
    def __init__(self, strategy: str = 'mean', fill_value: float = 0):
        self.strategy = strategy
        self.fill_value = fill_value
        self.imputer = None
        
    def fit(self, X, y=None):
        if self.strategy == 'zero':
            self.imputer = SimpleImputer(strategy='constant', fill_value=self.fill_value)
        elif self.strategy == 'mode':
            self.imputer = SimpleImputer(strategy='most_frequent')
        else:
            self.imputer = SimpleImputer(strategy=self.strategy)
            
        self.imputer.fit(X)
        return self

    def transform(self, X):
        if self.imputer is None:
            raise RuntimeError("Imputer has not been fitted yet.")
        return self.imputer.transform(X)

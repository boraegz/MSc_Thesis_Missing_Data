import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

class DataSimulator:
    def __init__(self, n_samples=1000, n_features=10, random_state=42):
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state

    def simulate_credit_data(self):
        """Simulate a synthetic credit dataset."""
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=8,
            n_redundant=2,
            random_state=self.random_state
        )
        columns = [f'feature_{i}' for i in range(self.n_features)] + ['target']
        data = pd.DataFrame(np.hstack([X, y.reshape(-1, 1)]), columns=columns)
        return data

    def introduce_missingness(self, data, mechanism='MCAR', missing_proportion=0.2, missing_col='feature_0'):
        """
        Introduce missingness into the dataset.
        Mechanisms: MCAR, MAR, MNAR.
        """
        # Set fixed seed for reproducibility
        np.random.seed(42)  # Using a fixed seed value
        
        if mechanism == 'MCAR':
            # Missing completely at random
            mask = np.random.rand(len(data)) < missing_proportion
            data.loc[mask, missing_col] = np.nan

        elif mechanism == 'MAR':
            # Missing at random (dependent on another feature)
            dependent_col = 'feature_1'
            threshold = np.percentile(data[dependent_col], (1 - missing_proportion) * 100)
            mask = data[dependent_col] > threshold
            data.loc[mask, missing_col] = np.nan

        elif mechanism == 'MNAR':
            # Missing not at random (dependent on itself)
            threshold = np.percentile(data[missing_col], (1 - missing_proportion) * 100)
            mask = data[missing_col] > threshold
            data.loc[mask, missing_col] = np.nan

        return data
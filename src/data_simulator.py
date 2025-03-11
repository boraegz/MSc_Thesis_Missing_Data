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
        For MNAR, missingness is introduced in the target variable based on its own values.
        For MCAR and MAR, missingness is introduced in the specified feature column.
        """
        # Create a copy of the data to avoid modifying the original
        data = data.copy()
        
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
            # Missing not at random (dependent on target values)
            # For binary target, probability of being missing depends on the value
            prob_missing_if_0 = 0.1  # 10% chance of missing if target = 0
            prob_missing_if_1 = 0.3  # 30% chance of missing if target = 1
            
            # Generate random numbers for comparison
            random_nums = np.random.rand(len(data))
            
            # Create mask where probability of being missing depends on the target value
            mask = np.where(
                data['target'] == 1,
                random_nums < prob_missing_if_1,  # Higher probability of missing for 1s
                random_nums < prob_missing_if_0   # Lower probability of missing for 0s
            )
            
            # Apply the mask
            data.loc[mask, 'target'] = np.nan

        return data
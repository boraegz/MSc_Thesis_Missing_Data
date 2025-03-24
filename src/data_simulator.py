import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

class DataSimulator:
    def __init__(self, n_samples=1000, n_features=10, random_state=42):
        """
        Initialize the DataSimulator class.

        Parameters:
        -----------
        n_samples : int
            Number of samples in the dataset.
        n_features : int
            Number of features (excluding target).
        random_state : int
            Seed for reproducibility.
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state
        np.random.seed(random_state)  # Ensure reproducibility

    def simulate_credit_data(self):
        """Generate a synthetic credit dataset with numeric features."""
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=8,
            n_redundant=2,
            random_state=self.random_state
        )
        columns = [f'feature_{i}' for i in range(self.n_features)] + ['target']
        data = pd.DataFrame(np.hstack([X, y.reshape(-1, 1)]), columns=columns)

        # Convert all numeric values to absolute values (since they represent credit data)
        data.iloc[:, :-1] = data.iloc[:, :-1].abs()
        return data

    def introduce_missingness(self, data, mechanism='MCAR', missing_proportion=0.2, missing_col='feature_0'):
        """
        Introduce missing values into the dataset using different missing data mechanisms.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset.
        mechanism : str
            The missing data mechanism ('MCAR', 'MAR', 'MNAR').
        missing_proportion : float
            Proportion of missing values to introduce.
        missing_col : str
            Column affected by missingness (for MCAR and MAR).

        Returns:
        --------
        pd.DataFrame
            Dataset with missing values introduced.
        """
        data = data.copy()

        if mechanism == 'MCAR':
            # Missing Completely at Random
            mask = np.random.rand(len(data)) < missing_proportion
            data.loc[mask, missing_col] = np.nan

        elif mechanism == 'MAR':
            # Missing at Random (dependent on another feature)
            dependent_col = 'feature_1'
            threshold = np.percentile(data[dependent_col], (1 - missing_proportion) * 100)
            mask = data[dependent_col] > threshold
            data.loc[mask, missing_col] = np.nan

        elif mechanism == 'MNAR':
            # Missing Not at Random (dependent on target values)
            prob_missing_if_0 = 0.1  # 10% chance of missing if target = 0
            prob_missing_if_1 = 0.3  # 30% chance of missing if target = 1
            random_nums = np.random.rand(len(data))

            # Apply probability-based missingness on target column
            mask = np.where(
                data['target'] == 1, 
                random_nums < prob_missing_if_1,
                random_nums < prob_missing_if_0
            )
            data.loc[mask, 'target'] = np.nan

        else:
            raise ValueError("Invalid mechanism! Choose from 'MCAR', 'MAR', 'MNAR'.")

        return data
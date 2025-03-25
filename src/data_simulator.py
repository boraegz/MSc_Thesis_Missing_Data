import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

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
        """
        Simulate synthetic credit data by generating two groups (good and bad applicants)
        using different Gaussian distributions. This method mimics the idea of having different
        distributions for accepted (good risk) and rejected (bad risk) applicants.

        Returns:
        --------
        pd.DataFrame
            A dataframe containing features and a target column.
        """
        # Define sample sizes for good and bad applicants (e.g., 60% good, 40% bad)
        n_good = int(self.n_samples * 0.6)
        n_bad = self.n_samples - n_good
        
        # Generate good applicants with lower risk using a normal distribution
        # (mean = 0.5, standard deviation = 1.0)
        X_good = np.random.normal(loc=0.5, scale=1.0, size=(n_good, self.n_features))
        
        # Generate bad applicants with higher risk using a normal distribution
        # (mean = 2.0, standard deviation = 1.5)
        X_bad = np.random.normal(loc=2.0, scale=1.5, size=(n_bad, self.n_features))
        
        # Concatenate the feature matrices and create the target vector (0 for good, 1 for bad)
        X = np.vstack([X_good, X_bad])
        y = np.hstack([np.zeros(n_good), np.ones(n_bad)])
        
        # Create a DataFrame and convert all feature values to absolute values
        columns = [f'feature_{i}' for i in range(self.n_features)]
        data = pd.DataFrame(X, columns=columns)
        data[columns] = data[columns].abs()
        data['target'] = y
        
        return data

    def acceptance_loop(self, data, threshold=1.0):
        """
        Simulate a simple acceptance loop similar to a credit scoring process:
        - If a sample's 'feature_0' value is below a given threshold, the application is accepted.
        - Otherwise, it is rejected.
        
        Note: In a real-world scenario, a scoring model output (e.g., probability of default)
              would be used for this decision.

        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset.
        threshold : float
            The threshold value for 'feature_0' to decide acceptance.

        Returns:
        --------
        pd.DataFrame
            The dataset with an additional 'accepted' column indicating acceptance status.
        """
        data = data.copy()
        # Accept the application if 'feature_0' is less than the threshold
        data['accepted'] = data['feature_0'] < threshold  
        return data

    def introduce_missingness(self, data, mechanism='MCAR', missing_proportion=0.2, missing_col='feature_0'):
        """
        Introduce missing values into the dataset based on different missing data mechanisms.

        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset.
        mechanism : str
            The missing data mechanism ('MCAR', 'MAR', 'MNAR').
        missing_proportion : float
            Proportion of missing values to introduce (used in MCAR and MAR).
        missing_col : str
            Column to be affected by missingness (for MCAR and MAR).

        Returns:
        --------
        pd.DataFrame
            Dataset with missing values introduced.
        """
        data = data.copy()

        if mechanism == 'MCAR':
            # Missing Completely at Random: randomly select rows to set the specified column to NaN
            mask = np.random.rand(len(data)) < missing_proportion
            data.loc[mask, missing_col] = np.nan

        elif mechanism == 'MAR':
            # Missing At Random: missingness in missing_col depends on another observed feature (e.g., feature_1)
            dependent_col = 'feature_1'
            threshold_val = np.percentile(data[dependent_col], (1 - missing_proportion) * 100)
            mask = data[dependent_col] > threshold_val
            data.loc[mask, missing_col] = np.nan

        elif mechanism == 'MNAR':
            # Missing Not At Random: missingness in the target variable is linked to both an observed feature (feature_1)
            # and the target value itself.
            # For example:
            # - If feature_1 is below 1 (indicating high risk) and target == 1 (default), 
            #   assign a higher probability (e.g., 30%) for target to be missing.
            # - If feature_1 is below 1 and target == 0, assign a lower probability (e.g., 10%).
            # - If feature_1 is 1 or above, do not induce missingness.
            # Note: This simulates a situation where missingness is not entirely random,
            # but related to the unobserved risk (target) in high-risk regions.
            mask = np.zeros(len(data), dtype=bool)
            # Process only rows where feature_1 < 1
            condition = data['feature_1'] < 1
            # For high risk (target == 1), missing probability is higher
            high_risk = (data['target'] == 1) & condition
            mask[high_risk] = np.random.rand(np.sum(high_risk)) < 0.3
            # For low risk (target == 0) in high-risk region, missing probability is lower
            low_risk = (data['target'] == 0) & condition
            mask[low_risk] = np.random.rand(np.sum(low_risk)) < 0.1
            # For rows where feature_1 >= 1, no missingness is introduced (target remains observed)
            data.loc[mask, 'target'] = np.nan

        else:
            raise ValueError("Invalid mechanism! Choose from 'MCAR', 'MAR', 'MNAR'.")

        return data
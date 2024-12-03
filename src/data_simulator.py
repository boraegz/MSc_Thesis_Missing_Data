import pandas as pd
import numpy as np

class DataSimulator:
    def __init__(self, num_samples=1000, num_features=10):
        """
        Initialize the simulator with the number of samples and features.
        Args:
            num_samples (int): Number of samples in the dataset.
            num_features (int): Number of features (columns) in the dataset.
        """
        self.num_samples = num_samples
        self.num_features = num_features
    
    def generate_data(self):
        """
        Generate synthetic data for credit scoring.
        Returns:
            pd.DataFrame: A DataFrame with simulated credit data.
        """
        np.random.seed(42)
        data = np.random.rand(self.num_samples, self.num_features)
        df = pd.DataFrame(data, columns=[f'feature_{i+1}' for i in range(self.num_features)])
        # Adding a binary repayment label
        df['Repayment_Label'] = np.random.choice([0, 1], size=self.num_samples, p=[0.3, 0.7])
        return df

    def introduce_missingness(self, data, missingness_type="MCAR", missing_rate=0.2):
        """
        Introduce missingness into the data based on the chosen missingness type.
        Args:
            data (pd.DataFrame): The data where missingness will be introduced.
            missingness_type (str): Type of missingness ('MCAR', 'MAR', 'MNAR').
            missing_rate (float): Percentage of missing values.
        Returns:
            pd.DataFrame: The data with missing values.
        """
        if missingness_type == "MCAR":
            missing_indices = np.random.rand(len(data)) < missing_rate
            data.loc[missing_indices, 'Repayment_Label'] = np.nan
        elif missingness_type == "MAR":
            missing_indices = np.random.rand(len(data)) < missing_rate
            data.loc[missing_indices, 'Repayment_Label'] = np.nan
        elif missingness_type == "MNAR":
            data = self.mnar_missingness(data, missing_rate)
        
        return data

    def mnar_missingness(self, data, missing_rate):
        """
        Apply MNAR (Missing Not at Random) mechanism for introducing missingness.
        Args:
            data (pd.DataFrame): The dataset where missingness will be introduced.
            missing_rate (float): Percentage of missingness.
        Returns:
            pd.DataFrame: The dataset with MNAR missingness.
        """
        missing_indices = np.random.rand(len(data)) < missing_rate
        data.loc[missing_indices, 'Repayment_Label'] = np.nan
        return data

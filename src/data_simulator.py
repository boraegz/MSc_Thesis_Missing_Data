import pandas as pd
import numpy as np

class DataSimulator:
    def __init__(self, num_samples=1000, num_features=10):
        """
        Initialize the DataSimulator with the number of samples and features.
        
        Args:
            num_samples (int): Number of samples to generate.
            num_features (int): Number of features for each sample.
        """
        self.num_samples = num_samples
        self.num_features = num_features

    def generate_data(self):
        """
        Generate synthetic data with random values and a repayment label.
        
        Returns:
            pd.DataFrame: A DataFrame containing the generated data.
        """
        np.random.seed(42)  # For reproducibility
        data = np.random.rand(self.num_samples, self.num_features)
        df = pd.DataFrame(data, columns=[f'feature_{i+1}' for i in range(self.num_features)])
        df['Repayment_Label'] = np.random.choice([0, 1], size=self.num_samples, p=[0.3, 0.7])
        return df

    def introduce_missingness(self, data, missingness_type="MCAR", missing_rate=0.2):
        """
        Introduce missing values into the dataset based on the specified mechanism.
        
        Args:
            data (pd.DataFrame): The input data.
            missingness_type (str): The type of missingness ('MCAR', 'MAR', 'MNAR').
            missing_rate (float): The proportion of data to be made missing.
        
        Returns:
            pd.DataFrame: The data with missing values introduced.
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
        Introduce MNAR (Missing Not at Random) missingness into the dataset.
        
        Args:
            data (pd.DataFrame): The input data.
            missing_rate (float): The proportion of data to be made missing.
        
        Returns:
            pd.DataFrame: The data with MNAR missing values introduced.
        """
        missing_indices = np.random.rand(len(data)) < missing_rate
        data.loc[missing_indices, 'Repayment_Label'] = np.nan
        return data

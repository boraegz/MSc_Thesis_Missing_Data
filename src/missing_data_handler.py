import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Union
import logging

class MissingDataHandler:
    """
    A class to handle missing data using various techniques:
    - Mean Imputation
    - Iterative Imputation
    - Heckman Correction (for MNAR data)
    - BASL (Bias-Aware Self-Learning)

    Examples:
    --------
    >>> import pandas as pd
    >>> handler = MissingDataHandler()
    >>> df = pd.DataFrame({'A': [1, None, 3], 'B': [4, 5, None]})
    >>> 
    >>> # Mean imputation
    >>> df_mean = handler.mean_imputation(df, 'A')
    >>> 
    >>> # Iterative imputation
    >>> df_iter = handler.iterative_imputation(df, ['A', 'B'])
    """

    def __init__(self):
        """
        Initialize the MissingDataHandler with a fixed random seed for reproducibility.
        """
        self.logger = logging.getLogger(__name__)
        self.random_state = 42  # Fixed random seed

    def mean_imputation(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Perform mean imputation for missing values in a specified column.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset with missing values.
        column : str
            The column with missing values to impute.

        Returns:
        --------
        pd.DataFrame
            The dataset with missing values imputed.

        Examples:
        --------
        >>> df = pd.DataFrame({'A': [1, None, 3]})
        >>> handler = MissingDataHandler()
        >>> df_imputed = handler.mean_imputation(df, 'A')
        """
        # Input validation
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not isinstance(column, str):
            raise TypeError("column must be a string")
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in the dataset")
        if data.empty:
            raise ValueError("Input DataFrame is empty")
        if data[column].isna().all():
            raise ValueError(f"Column '{column}' contains all missing values")

        imputer = SimpleImputer(strategy='mean')
        data[column] = imputer.fit_transform(data[[column]])
        return data

    def iterative_imputation(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Perform iterative imputation for missing values in specified columns.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset with missing values.
        columns : list of str
            The columns with missing values to impute.

        Returns:
        --------
        pd.DataFrame
            The dataset with missing values imputed.

        Examples:
        --------
        >>> df = pd.DataFrame({'A': [1, None, 3], 'B': [4, 5, None]})
        >>> handler = MissingDataHandler()
        >>> df_imputed = handler.iterative_imputation(df, ['A', 'B'])
        """
        # Input validation
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not isinstance(columns, list):
            raise TypeError("columns must be a list of column names")
        if data.empty:
            raise ValueError("Input DataFrame is empty")
        
        for col in columns:
            if not isinstance(col, str):
                raise TypeError(f"Column name '{col}' must be a string")
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in the dataset")
            if data[col].isna().all():
                raise ValueError(f"Column '{col}' contains all missing values")

        imputer = IterativeImputer(max_iter=10, random_state=self.random_state)
        data[columns] = imputer.fit_transform(data[columns])
        return data

    def heckman_correction(self, data: pd.DataFrame, missing_col: str, selection_col: str) -> pd.DataFrame:
        """
        Perform Heckman correction for MNAR (Missing Not At Random) data.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset with missing values.
        missing_col : str
            The column with MNAR missing values.
        selection_col : str
            The column used for the selection model (e.g., a feature correlated with missingness).

        Returns:
        --------
        pd.DataFrame
            The dataset with missing values corrected using Heckman's method.

        Examples:
        --------
        >>> df = pd.DataFrame({
        ...     'income': [50000, None, 75000, None],
        ...     'education': [16, 12, 18, 14]
        ... })
        >>> handler = MissingDataHandler()
        >>> df_corrected = handler.heckman_correction(df, 'income', 'education')
        """
        # Input validation
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not isinstance(missing_col, str) or not isinstance(selection_col, str):
            raise TypeError("Column names must be strings")
        if missing_col not in data.columns or selection_col not in data.columns:
            raise ValueError(f"Columns '{missing_col}' or '{selection_col}' not found in the dataset")
        if data.empty:
            raise ValueError("Input DataFrame is empty")
        if data[selection_col].isna().any():
            raise ValueError(f"Selection column '{selection_col}' contains missing values")

        # Step 1: Probit model for selection (predict missingness)
        missing_mask = data[missing_col].isna().astype(int)
        
        # Add epsilon to avoid division by zero
        epsilon = 1e-10
        probit = data[selection_col].values.reshape(-1, 1)
        inv_mills = norm.pdf(probit) / (norm.cdf(probit) + epsilon)

        # Step 3: Add inverse Mills ratio to the dataset
        data['inv_mills'] = inv_mills

        # Step 4: Fit the outcome model (predict missing values)
        outcome_model = LinearRegression()
        outcome_model.fit(data[['inv_mills', selection_col]], data[missing_col].fillna(0))

        # Step 5: Impute missing values
        data[missing_col] = outcome_model.predict(data[['inv_mills', selection_col]])

        # Drop the inverse Mills ratio column
        data.drop(columns=['inv_mills'], inplace=True)

        return data

    def basl_method(self, data: pd.DataFrame, missing_col: str, n_iter: int = 10) -> pd.DataFrame:
        """
        Perform Bias-Aware Self-Learning (BASL) for MNAR data.
        For binary target variables, ensures predictions are rounded to 0 or 1.

        Parameters:
        -----------
        data : pd.DataFrame
            The dataset with missing values.
        missing_col : str
            The column with MNAR missing values.
        n_iter : int, optional
            The number of iterations for self-learning (default: 10).

        Returns:
        --------
        pd.DataFrame
            The dataset with missing values imputed using BASL.
        """
        # Input validation
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if not isinstance(missing_col, str):
            raise TypeError("missing_col must be a string")
        if not isinstance(n_iter, int):
            raise TypeError("n_iter must be an integer")
        if n_iter <= 0:
            raise ValueError("n_iter must be positive")
        if missing_col not in data.columns:
            raise ValueError(f"Column '{missing_col}' not found in the dataset")
        if data.empty:
            raise ValueError("Input DataFrame is empty")
        if data[missing_col].isna().all():
            raise ValueError(f"Column '{missing_col}' contains all missing values")

        # Create a copy to avoid modifying the original data
        data_copy = data.copy()
        
        # Handle categorical variables
        categorical_columns = data_copy.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            data_copy = pd.get_dummies(data_copy, columns=categorical_columns)

        # Initialize imputed values with mean imputation
        initial_imputer = SimpleImputer(strategy='mean')
        data_copy[missing_col] = initial_imputer.fit_transform(data_copy[[missing_col]])

        # Handle missing values in other columns
        features = data_copy.drop(columns=[missing_col])
        feature_imputer = SimpleImputer(strategy='mean')
        features = pd.DataFrame(feature_imputer.fit_transform(features), columns=features.columns)

        # Add convergence tracking
        prev_values = None
        tolerance = 1e-6
        
        # Self-learning loop with convergence check
        np.random.seed(self.random_state)
        for iteration in range(n_iter):
            model = LinearRegression()
            model.fit(features, data_copy[missing_col])
            new_values = model.predict(features)
            
            # Check convergence
            if prev_values is not None:
                change = np.mean(np.abs(new_values - prev_values))
                self.logger.info(f"Iteration {iteration + 1}, mean change: {change:.6f}")
                if change < tolerance:
                    self.logger.info(f"Converged after {iteration + 1} iterations")
                    break
                    
            data_copy[missing_col] = new_values
            prev_values = new_values

        # Round the final predictions to 0 or 1 for binary target
        data_copy[missing_col] = np.round(data_copy[missing_col]).astype(int)

        # Copy the imputed values back to original data
        data[missing_col] = data_copy[missing_col]
        return data
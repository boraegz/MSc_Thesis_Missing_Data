import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the credit scoring dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw input data
        
    Returns:
    --------
    pd.DataFrame
        Preprocessed data
    """
    # Create a copy to avoid modifying the original data
    df = data.copy()
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Handle missing values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # Handle missing values in categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    
    # Remove outliers using IQR method
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
    
    logger.info("Data preprocessing completed")
    return df

def plot_missingness(data: pd.DataFrame) -> None:
    """
    Visualize missing values in the dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()
    
    # Missing values percentage
    missing_percentages = (data.isnull().sum() / len(data)) * 100
    missing_percentages = missing_percentages[missing_percentages > 0].sort_values(ascending=True)
    
    if len(missing_percentages) > 0:
        plt.figure(figsize=(10, 6))
        missing_percentages.plot(kind='barh')
        plt.title('Percentage of Missing Values by Feature')
        plt.xlabel('Percentage')
        plt.show()

def plot_feature_distributions(data: pd.DataFrame, 
                             numeric_cols: List[str] = None) -> None:
    """
    Plot distributions of numeric features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    numeric_cols : list of str, optional
        List of numeric columns to plot (default: None, uses all numeric columns)
    """
    if numeric_cols is None:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    axes = axes.ravel()
    
    for idx, col in enumerate(numeric_cols):
        sns.histplot(data=data, x=col, ax=axes[idx])
        axes[idx].set_title(f'Distribution of {col}')
        
    # Hide empty subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)
        
    plt.tight_layout()
    plt.show()

def create_correlation_matrix(data: pd.DataFrame) -> None:
    """
    Create and plot correlation matrix for numeric features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    """
    numeric_data = data.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.show()

def split_data(X: pd.DataFrame, y: pd.Series, 
               test_size: float = 0.2, 
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split the dataset into training and testing sets.
    
    Parameters:
    -----------
    X : pd.DataFrame
        Feature set
    y : pd.Series
        Target variable
    test_size : float, optional
        Proportion of dataset to include in the test split (default: 0.2)
    random_state : int, optional
        Random state for reproducibility (default: 42)
        
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    logger.info(f"Data split completed. Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def visualize_results(evaluation_results, title="Model Evaluation Results"):
    """
    Visualize the model evaluation results (e.g., Accuracy, AUC) using a barplot.
    Args:
        evaluation_results (dict): Dictionary containing model evaluation metrics.
        title (str): Title for the plot.
    """
    sns.barplot(x=list(evaluation_results.keys()), y=list(evaluation_results.values()))
    plt.title(title)
    plt.xlabel("Metrics")
    plt.ylabel("Scores")
    plt.show()
    
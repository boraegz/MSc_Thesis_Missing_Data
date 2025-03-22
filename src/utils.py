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

def plot_missingness(data: pd.DataFrame, dataset_name: str = "Dataset") -> None:
    """
    Visualize missing values in the dataset.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    dataset_name : str
        Name of the dataset for plot titles
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.title(f'Missing Values Heatmap - {dataset_name}')
    plt.show()
    
    # Missing values percentage
    missing_percentages = (data.isnull().sum() / len(data)) * 100
    missing_percentages = missing_percentages[missing_percentages > 0].sort_values(ascending=True)
    
    if len(missing_percentages) > 0:
        plt.figure(figsize=(10, 6))
        missing_percentages.plot(kind='barh')
        plt.title(f'Percentage of Missing Values by Feature - {dataset_name}')
        plt.xlabel('Percentage')
        plt.show()

def plot_feature_distributions(data: pd.DataFrame, 
                             dataset_name: str = "Dataset",
                             numeric_cols: List[str] = None) -> None:
    """
    Plot distributions of features with appropriate visualization for different data types.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    dataset_name : str
        Name of the dataset for plot titles
    numeric_cols : list of str, optional
        List of numeric columns to plot (default: None, uses all numeric columns)
    """
    if numeric_cols is None:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
    
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
    fig.suptitle(f'Feature Distributions - {dataset_name}', fontsize=16, y=1.02)
    axes = axes.ravel()
    
    for idx, col in enumerate(numeric_cols):
        unique_values = data[col].nunique()
        
        if unique_values <= 2:  # Binary feature
            # Create count plot for binary features
            sns.countplot(data=data, x=col, ax=axes[idx])
            axes[idx].set_title(f'Distribution of {col} (Binary)')
            
            # Add percentage labels on top of bars
            total = len(data[col])
            for p in axes[idx].patches:
                percentage = f'{100 * p.get_height()/total:.1f}%'
                axes[idx].annotate(percentage, (p.get_x() + p.get_width()/2., p.get_height()),
                                 ha='center', va='bottom')
            
            # Force x-axis to show only 0 and 1
            axes[idx].set_xticks([0, 1])
            axes[idx].set_xticklabels(['0', '1'])
            
        elif unique_values <= 10:  # Categorical feature with few unique values
            # Create count plot for categorical features
            sns.countplot(data=data, x=col, ax=axes[idx])
            axes[idx].set_title(f'Distribution of {col} (Categorical)')
            
            # Add percentage labels on top of bars
            total = len(data[col])
            for p in axes[idx].patches:
                percentage = f'{100 * p.get_height()/total:.1f}%'
                axes[idx].annotate(percentage, (p.get_x() + p.get_width()/2., p.get_height()),
                                 ha='center', va='bottom')
            
        else:  # Continuous feature
            # Calculate optimal number of bins using Freedman-Diaconis rule
            q75, q25 = np.percentile(data[col].dropna(), [75, 25])
            iqr = q75 - q25
            bin_width = 2 * iqr / (len(data[col]) ** (1/3))
            n_bins = int(np.ceil((data[col].max() - data[col].min()) / bin_width))
            n_bins = min(max(n_bins, 30), 100)  # Keep bins between 30 and 100
            
            # Create histogram with KDE and adaptive bins
            sns.histplot(data=data, x=col, ax=axes[idx], kde=True, bins=n_bins, stat='count')
            axes[idx].set_title(f'Distribution of {col} (Continuous)')
        
        # Add statistics as text
        stats_text = f'Unique values: {unique_values}\n'
        stats_text += f'Mean: {data[col].mean():.2f}\n'
        stats_text += f'Median: {data[col].median():.2f}\n'
        stats_text += f'Std: {data[col].std():.2f}\n'
        stats_text += f'Missing: {data[col].isnull().sum()}'
        
        axes[idx].text(0.02, 0.98, stats_text,
                      transform=axes[idx].transAxes,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Customize axes
        axes[idx].set_xlabel(f'{col} Values')
        axes[idx].set_ylabel('Count')
        axes[idx].grid(True, alpha=0.3)
        
        # Format x-axis to show actual values
        if unique_values > 2:  # Only format x-axis for non-binary features
            axes[idx].xaxis.set_major_locator(plt.MaxNLocator(10))  # Show 10 major ticks
            axes[idx].tick_params(axis='x', rotation=45)
        
        # Add value range to title for continuous features
        if unique_values > 10:
            value_range = f'Range: [{data[col].min():.2f}, {data[col].max():.2f}]'
            axes[idx].set_title(f'Distribution of {col} (Continuous)\n{value_range}')
        
    # Hide empty subplots
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)
        
    plt.tight_layout()
    plt.show()

def create_correlation_matrix(data: pd.DataFrame, dataset_name: str = "Dataset") -> None:
    """
    Create and plot correlation matrix for numeric features.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Input data
    dataset_name : str
        Name of the dataset for plot title
    """
    numeric_data = data.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(f'Correlation Matrix - {dataset_name}')
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

def dataset_statistics(df, name):
    total_entries = len(df)
    missing_values = df.isnull().sum().sum()
    missing_percentage = (missing_values * 100 / total_entries) if df.size > 0 else 0

    print(f"\n{name} Dataset:")
    print("-" * 50)
    print(f"Total entries: {total_entries}")
    print(f"Total missing values: {missing_values}")
    print(f"Missing value percentage: {missing_percentage:.2f}%")
    print(f"Mean of numeric columns:\n{df.mean(numeric_only=True)}\n")
    print(f"Median of numeric columns:\n{df.median(numeric_only=True)}\n")
    
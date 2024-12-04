import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def split_data(X, y, test_size=0.3, random_state=42):
    """
    Split the dataset into training and testing sets.
    Args:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Labels.
        test_size (float): Fraction of the data to use as test set.
        random_state (int): Random seed for reproducibility.
    Returns:
        X_train, X_test, y_train, y_test (tuple): Split data.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
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
    
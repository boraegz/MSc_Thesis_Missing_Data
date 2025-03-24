import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import logging


class CreditScoringModel:
    """
    A class to handle credit scoring model training, prediction, and evaluation.
    """

    def __init__(self, random_state: int = 42, class_weight: str = 'balanced'):
        """
        Initialize the CreditScoringModel.

        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility (default: 42)
        class_weight : str
            Class weight for handling imbalanced classes, default is 'balanced'
        """
        self.logger = logging.getLogger(__name__)
        self.random_state = random_state
        self.model = RandomForestClassifier(random_state=self.random_state, class_weight=class_weight)
        self.scaler = StandardScaler()

    def prepare_data(self, data: pd.DataFrame, target: str, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for model training.

        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset
        target : str
            The name of the target column
        test_size : float
            Proportion of dataset to include in the test split (default: 0.2)

        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in dataset")

        # Separate features and target
        X = data.drop(columns=[target])
        y = data[target]

        # Handle categorical variables
        X = pd.get_dummies(X, drop_first=True)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # Scale numeric features
        X_train = pd.DataFrame(self.scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

        return X_train, X_test, y_train, y_test

    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> 'CreditScoringModel':
        """
        Train the model.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target

        Returns:
        --------
        CreditScoringModel
            The trained model instance
        """
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.

        Parameters:
        -----------
        X : pd.DataFrame
            Features to make predictions on

        Returns:
        --------
        np.ndarray
            Predicted classes
        """
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability estimates.

        Parameters:
        -----------
        X : pd.DataFrame
            Features to make predictions on

        Returns:
        --------
        np.ndarray
            Probability estimates for each class
        """
        return self.model.predict_proba(X)

    def evaluate_model(self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """
        Trains the model, makes predictions, and evaluates its performance.

        Parameters:
        -----------
        X_train : pd.DataFrame
            The training features
        X_test : pd.DataFrame
            The test features
        y_train : pd.Series
            The training target
        y_test : pd.Series
            The test target

        Returns:
        --------
        dict
            A dictionary containing performance metrics and confusion matrix
        """
        self.train(X_train, y_train)

        # Predictions
        predictions = self.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, predictions)

        # Classification report
        class_report = classification_report(y_test, predictions, output_dict=True)

        # Convert classification report to DataFrame
        class_report_df = pd.DataFrame(class_report).transpose()

        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)

        # Visualize confusion matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Loan', 'Loan'], yticklabels=['No Loan', 'Loan'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.show()

        # Create summary table with only the required metrics
        metrics = ["accuracy", "precision", "recall", "f1-score"]

        # Extract class labels properly (handle both integer and string keys)
        valid_classes = [cls for cls in class_report.keys() if cls.replace('.', '', 1).isdigit()]

        # Compute metrics correctly
        metric_values = [accuracy]  # Start with accuracy
        for metric in ["precision", "recall", "f1-score"]:
            values = [class_report[cls][metric] for cls in valid_classes if metric in class_report[cls]]
            metric_values.append(np.mean(values) if values else 0)

        # Construct summary DataFrame
        summary_df = pd.DataFrame({
            'Metric': ["accuracy", "precision", "recall", "f1-score"],
            'Value': metric_values
        })

        print("\nModel Performance Summary:")
        print(summary_df)

        return {
            "accuracy": accuracy,
            "classification_report": class_report,
            "confusion_matrix": cm
        }
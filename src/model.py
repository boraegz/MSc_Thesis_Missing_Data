import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from typing import Tuple, Dict, Any

class CreditScoringModel:
    """
    A class to handle credit scoring model training and prediction.
    
    Examples:
    --------
    >>> model = CreditScoringModel()
    >>> X_train, X_test, y_train, y_test = model.prepare_data(df, target='loan_status')
    >>> model.train(X_train, y_train)
    >>> predictions = model.predict(X_test)
    """
    
    def __init__(self, random_state: int = 42, class_weight: str = 'balanced'):
        """
        Initialize the CreditScoringModel.
        
        Parameters:
        -----------
        random_state : int, optional
            Random seed for reproducibility (default: 42)
        class_weight : str, optional
            Class weight for handling imbalanced classes, default is 'balanced'
        """
        self.logger = logging.getLogger(__name__)
        self.random_state = random_state
        self.model = RandomForestClassifier(random_state=self.random_state, class_weight=class_weight)
        self.scaler = StandardScaler()
        
    def prepare_data(self, data: pd.DataFrame, target: str, 
                    test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for model training.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The input dataset
        target : str
            The name of the target column
        test_size : float, optional
            Proportion of dataset to include in the test split (default: 0.2)
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")
        if target not in data.columns:
            raise ValueError(f"Target column '{target}' not found in dataset")
            
        # Separate features and target
        X = data.drop(columns=[target])
        y = data[target]
        
        # Handle categorical variables
        X = pd.get_dummies(X, drop_first=True)  # drop_first to avoid multicollinearity
        
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Scale the features (for continuous variables)
        X_train = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
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
        self.logger.info("Training model...")
        self.model.fit(X_train, y_train)
        self.logger.info("Model training completed")
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
        Make probability predictions using the trained model.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features to make predictions on
            
        Returns:
        --------
        np.ndarray
            Predicted probabilities for each class
        """
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
        --------
        dict
            Dictionary mapping feature names to importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise AttributeError("Model has not been trained yet")
            
        return dict(zip(self.model.feature_names_in_, self.model.feature_importances_))

    
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
        # Train the model
        self.train(X_train, y_train)
        
        # Predict on the test set
        predictions = self.predict(X_test)
        predictions_proba = self.predict_proba(X_test)
    
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        class_report = classification_report(y_test, predictions, output_dict=True)
        cm = confusion_matrix(y_test, predictions)
        
        # Store results
        results = {
            "accuracy": accuracy,
            "classification_report": class_report,
            "confusion_matrix": cm
        }
        
        # Optionally: display confusion matrix as heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Loan', 'Loan'], yticklabels=['No Loan', 'Loan'])
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.show()
        
        return results
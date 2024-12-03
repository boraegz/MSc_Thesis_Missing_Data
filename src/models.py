# src/models.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

class ModelTrainer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train_model(self, X_train, y_train):
        """
        Train a Random Forest model.
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
        Returns:
            model: The trained model.
        """
        self.model.fit(X_train, y_train)
        return self.model

    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the trained model.
        Args:
            model: The trained model.
            X_test (pd.DataFrame): Testing features.
            y_test (pd.Series): Testing labels.
        Returns:
            dict: Model evaluation metrics.
        """
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        return {'AUC': auc, 'Accuracy': accuracy}

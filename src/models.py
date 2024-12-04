# src/models.py

from sklearn.ensemble import RandomForestClassifier

class ModelTrainer:
    def __init__(self):
        """
        Initialize the ModelTrainer with a RandomForestClassifier.
        """
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train_model(self, X_train, y_train):
        """
        Train the RandomForest model using the provided training data.
        
        Args:
            X_train (pd.DataFrame): Training features.
            y_train (pd.Series): Training labels.
        
        Returns:
            model: The trained RandomForest model.
        """
        self.model.fit(X_train, y_train)
        return self.model

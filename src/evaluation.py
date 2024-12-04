from sklearn.metrics import roc_auc_score, accuracy_score

class Evaluation:
    def evaluate_model(self, model, X_test, y_test):
        """
        Evaluate the model using AUC and accuracy metrics.
        
        Args:
            model: The trained model to evaluate.
            X_test (pd.DataFrame): Test features.
            y_test (pd.Series): True labels for the test set.
        
        Returns:
            dict: A dictionary containing AUC and accuracy scores.
        """
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        return {'AUC': auc, 'Accuracy': accuracy}

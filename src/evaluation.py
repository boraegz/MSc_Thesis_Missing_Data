from sklearn.metrics import roc_auc_score, accuracy_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model using AUC and Accuracy metrics.
    Args:
        model: The trained model.
        X_test (pd.DataFrame): Testing features.
        y_test (pd.Series): Testing labels.
    Returns:
        dict: Evaluation metrics.
    """
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return {'AUC': auc, 'Accuracy': accuracy}

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    brier_score_loss, 
    accuracy_score, 
    f1_score,
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

class Evaluator:
    """
    Compute and store evaluation metrics for binary classification models.
    """
    def __init__(self):
        pass

    def compute_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute standard binary classification metrics.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities of positive class
            threshold: Threshold for converting probabilities to labels (default 0.5)
            
        Returns:
            Dictionary of metrics.
        """
        # Convert probabilities to hard predictions
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        metrics = {
            'auc_roc': roc_auc_score(y_true, y_pred_proba),
            'auc_pr': average_precision_score(y_true, y_pred_proba),
            'brier_score': brier_score_loss(y_true, y_pred_proba),
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
        }
        
        return metrics

    def plot_calibration_curve(self, y_true, y_pred_proba, name: str = 'Model'):
        """
        Plot reliability diagram (Calibration curve).
        """
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
        
        plt.figure(figsize=(8, 6))
        plt.plot(prob_pred, prob_true, marker='o', label=name)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title(f'Calibration Curve - {name}')
        plt.legend()
        plt.grid()
        plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from typing import Dict, Any, List
import logging

class ModelEvaluator:
    """
    A class to evaluate credit scoring models.
    
    Examples:
    --------
    >>> evaluator = ModelEvaluator()
    >>> metrics = evaluator.evaluate_model(y_true, y_pred)
    >>> evaluator.plot_roc_curve(y_true, y_pred_proba)
    """
    
    def __init__(self):
        """Initialize the ModelEvaluator."""
        self.logger = logging.getLogger(__name__)
        
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate various classification metrics.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        y_pred_proba : np.ndarray, optional
            Predicted probabilities for positive class
            
        Returns:
        --------
        dict
            Dictionary containing various metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            
        self.logger.info("Model evaluation completed")
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                            normalize: bool = True) -> None:
        """
        Plot confusion matrix.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
        normalize : bool, optional
            Whether to normalize the confusion matrix (default: True)
        """
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> None:
        """
        Plot ROC curve.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred_proba : np.ndarray
            Predicted probabilities for positive class
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
        
    def compare_models(self, results: Dict[str, Dict[str, float]], 
                      metrics: List[str] = None) -> None:
        """
        Compare different models using bar plots.
        
        Parameters:
        -----------
        results : dict
            Dictionary containing metrics for different models
        metrics : list of str, optional
            List of metrics to compare (default: None, uses all metrics)
        """
        if metrics is None:
            metrics = list(next(iter(results.values())).keys())
            
        n_metrics = len(metrics)
        n_models = len(results)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
            
        for i, metric in enumerate(metrics):
            values = [result[metric] for result in results.values()]
            axes[i].bar(results.keys(), values)
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_ylim(0, 1)
            axes[i].tick_params(axis='x', rotation=45)
            
        plt.tight_layout()
        plt.show()
        
    def generate_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Generate a classification report.
        
        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_pred : np.ndarray
            Predicted labels
            
        Returns:
        --------
        str
            Classification report as string
        """
        return classification_report(y_true, y_pred)

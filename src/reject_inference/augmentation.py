import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

class Augmentation(BaseEstimator, ClassifierMixin):
    """
    Implements Augmentation (also known as Parceling or Self-training) for Reject Inference.
    
    1. Train base_estimator on Accepted (Labeled) samples.
    2. Predict labels (or probabilities) for Rejected (Unlabeled) samples.
    3. Combine Accepted + Rejected (with inferred labels) into an Augmented dataset.
    """
    def __init__(self, base_estimator=None, soft_labels: bool = False):
        self.base_estimator = base_estimator if base_estimator is not None else LogisticRegression()
        self.soft_labels = soft_labels
        self.estimator_ = None
        
    def fit(self, X, y_observed, mask):
        """
        Fits the augmentation model on accepted data.
        
        Args:
            X: Features (n_samples, n_features)
            y_observed: Target with NaNs for rejected.
            mask: 1=Accepted, 0=Rejected
        """
        self.estimator_ = self.base_estimator
        
        X_acc = X[mask == 1]
        y_acc = y_observed[mask == 1]
        
        self.estimator_.fit(X_acc, y_acc)
        return self
        
    def transform(self, X, y_observed, mask):
        """
        Returns the Augmented Dataset (X, y_augmented).
        
        Args:
            X: Features
            y_observed: Target with NaNs
            mask: 1=Accepted, 0=Rejected
            
        Returns:
            X_aug: Same as X
            y_aug: y_observed with NaNs filled by model predictions
        """
        check_is_fitted(self.estimator_)
        
        X_rej = X[mask == 0]
        
        if len(X_rej) == 0:
            return X, y_observed
            
        # Predict labels for rejected
        if self.soft_labels:
            # For soft labels, we might use probabilities.
            # But standard classifiers expect valid y labels. 
            # This requires downstream models to handle soft targets, which is rare in standard sklearn.
            # Keeping it simple: Hard labels for now unless requested.
            y_rej_pred = self.estimator_.predict(X_rej)
        else:
            y_rej_pred = self.estimator_.predict(X_rej)
            
        y_aug = y_observed.copy()
        y_aug[mask == 0] = y_rej_pred
        
        return X, y_aug

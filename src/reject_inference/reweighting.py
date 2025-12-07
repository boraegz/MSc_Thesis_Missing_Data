import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator
from typing import Optional

class InverseProbabilityWeighter(BaseEstimator):
    """
    Implements Inverse Probability Weighting (IPW) for Reject Inference.
    
    1. Trains a model P(Accepted | X).
    2. Computes weights w = 1 / P(Accepted | X) for accepted samples.
    """
    def __init__(self, estimator=None):
        self.estimator = estimator if estimator is not None else LogisticRegression(solver='lbfgs')
        self.weights_ = None
        
    def fit(self, X, mask):
        """
        Fit the propensity model.
        
        Args:
            X: Features (n_samples, n_features)
            mask: Binary array (n_samples,), 1=Accepted/Observed, 0=Rejected/Missing
        """
        # Train classifier to predict mask (Acceptance)
        self.estimator.fit(X, mask)
        return self
        
    def get_weights(self, X):
        """
        Compute inverse probability weights for the input X.
        
        Returns:
            weights: Array of shape (n_samples,). 
                     For samples with high prob of acceptance, weight is close to 1.
                     For samples with low prob of acceptance (resembling rejects), weight is high.
        """
        # Predict P(Accepted=1 | X)
        probs = self.estimator.predict_proba(X)[:, 1]
        
        # Avoid division by zero
        probs = np.clip(probs, 1e-6, 1.0)
        
        weights = 1.0 / probs
        return weights

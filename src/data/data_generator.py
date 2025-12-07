import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional
from src.utils import set_seed

class DataGenerator:
    """
    Generates synthetic datasets for credit scoring reject inference.
    Simulates:
    1. Oracle data (Ground Truth)
    2. Missingness mechanisms (MCAR, MAR, MNAR) to simulate rejection.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.seed = config.get('random_state', 42)
        set_seed(self.seed)
        
    def generate_oracle_data(self) -> Dict[str, np.ndarray]:
        """
        Generates the complete 'oracle' dataset using sklearn.make_classification.
        
        Returns:
            Dict containing:
                'X': Features (n_samples, n_features)
                'y': Target (n_samples,) - 0: Repaid, 1: Default
        """
        ds_config = self.config['dataset']
        
        X, y = make_classification(
            n_samples=ds_config['n_samples'],
            n_features=ds_config['n_features'],
            n_informative=ds_config['n_informative'],
            n_redundant=ds_config['n_redundant'],
            n_repeated=ds_config['n_repeated'],
            n_classes=ds_config['n_classes'],
            weights=ds_config['weights'],
            flip_y=ds_config['noise'],
            random_state=self.seed
        )
        
        # Ensure y is binary 0/1
        return {'X': X, 'y': y}

    def introduce_missingness(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Applies missingness to the target variable 'y' to simulate rejected applicants.
        
        Args:
            data: Dictionary with 'X' and 'y'.
            
        Returns:
            Dictionary containing:
                'X': Features (Standardized)
                'y_oracle': Original full target lines
                'y_observed': Target with NaNs for rejected applicants
                'mask': Boolean mask (True = Observed/Accepted, False = Missing/Rejected)
        """
        X = data['X']
        y = data['y']
        
        miss_config = self.config['missingness']
        mechanism = miss_config['mechanism']
        missing_rate = miss_config['missing_rate']
        
        n_samples = len(y)
        
        # Standardize X for consistent mechanism behavior
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        if mechanism == 'mcar':
            # Missing Completely At Random
            # Randomly select subset to reject
            mask = np.random.rand(n_samples) > missing_rate
            
        elif mechanism == 'mar':
            # Missing At Random
            # Probability of missingness depends on observed X (e.g. first feature)
            # Higher feature value -> higher chance of being accepted (observed)
            logistic = 1 / (1 + np.exp(-(X_scaled[:, 0] + X_scaled[:, 1])))
            
            # Adjust threshold to match desired missing_rate
            # This is a simplification; for strict rates we might use quantiles
            threshold = np.percentile(logistic, missing_rate * 100)
            mask = logistic > threshold
            
        elif mechanism == 'mnar_latent':
            # Missing Not At Random (Latent)
            # Missingness depends on the unobserved target y (Defaults are more likely to be rejected?)
            # Or in credit scoring: we reject high-risk people. 
            # If our model is good, rejected people (missing y) have HIGH probability of default (y=1).
            
            # Let's say we reject people who look risky.
            # But true MNAR means it depends on Y itself.
            mask = np.ones(n_samples, dtype=bool)
            
            # Reject 80% of actual defaulters (y=1) -> severe MNAR
            # Reject 10% of good payers (y=0)
            
            prob_reject_given_default = missing_rate * 1.5 # Heuristic
            prob_reject_given_repay = missing_rate * 0.5   # Heuristic
            
            rand = np.random.rand(n_samples)
            for i in range(n_samples):
                if y[i] == 1: # Defaulter
                    if rand[i] < prob_reject_given_default:
                        mask[i] = False
                else: # Repayer
                    if rand[i] < prob_reject_given_repay:
                        mask[i] = False
                        
        elif mechanism == 'mnar_rejection':
            # This simulates a "Credit Policy" rejection.
            # We train a logistic regression on X to predict Y, then reject the lowest scores.
            # This is standard "Reject Inference" setup: The missingness depends on X (Strictly MAR? or MNAR?)
            # Actually, if selection is based purely on X, it is MAR. 
            # EXCEPT if there are unobserved variables Z that affect both Y and Selection.
            # For this simulation, we use a proxy score based on X.
            
            slope = miss_config.get('logistic_slope', 1.0)
            intercept = miss_config.get('logistic_intercept', 0.0)
            
            # Create a latent score based on informative features
            # config says first 10 are informative. Use first 5 for scoring.
            latent_score = np.sum(X_scaled[:, :5], axis=1) * slope + intercept + np.random.normal(0, 0.5, n_samples)
            
            # Reject the lowest scoring candidates (missing_rate %)
            threshold = np.percentile(latent_score, missing_rate * 100)
            mask = latent_score > threshold

        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")

        # Create observed y
        y_observed = y.copy().astype(float)
        y_observed[~mask] = np.nan
        
        return {
            'X': X_scaled,            # Return scaled features
            'y_oracle': y,
            'y_observed': y_observed,
            'mask': mask.astype(int)  # 1=Observed, 0=Missing
        }

if __name__ == "__main__":
    # Simple test block
    from src.utils import load_config
    try:
        cfg = load_config("configs/experiment_config.yaml")
        gen = DataGenerator(cfg)
        data = gen.generate_oracle_data()
        print(f"Oracle generated. Attributes: {list(data.keys())} Shape: {data['X'].shape}")
        
        data_miss = gen.introduce_missingness(data)
        print(f"Missingness applied ({cfg['missingness']['mechanism']}). Attributes: {list(data_miss.keys())}")
        
        n_missing = np.isnan(data_miss['y_observed']).sum()
        print(f"Missing labels: {n_missing} / {len(data['y'])} ({n_missing/len(data['y']):.2%})")
        
    except Exception as e:
        print(f"Error: {e}")

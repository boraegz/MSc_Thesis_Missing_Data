import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional
import time

# Import our modules
from src.data.data_generator import DataGenerator
from src.imputation.baseline_imputer import BaselineImputer
from src.imputation.mice_imputer import MiceImputer
from src.imputation.missforest_imputer import MissForestImputer
from src.reject_inference.reweighting import InverseProbabilityWeighter
from src.reject_inference.augmentation import Augmentation
from src.evaluation import Evaluator

class ExperimentRunner:
    """
    Orchestrates the running of experiments across different missingness mechanisms and correction methods.
    """
    def __init__(self, config: Dict):
        self.config = config
        self.evaluator = Evaluator()
        self.results = []
        
    def run_pipeline(self, mechanism: str, missing_rate: float, method: str):
        """
        Runs a single experiment pipeline.
        
        Args:
            mechanism: 'mcar', 'mar', 'mnar_rejection'
            missing_rate: float
            method: 'complete_case', 'mean', 'mice', 'missforest', 'reweighting', 'augmentation'
        """
        # 1. Generate Data (Fresh seed for each run? Or same data?)
        # Ideally, we want same data for all methods to be comparable.
        # But we might run multiple seeds later.
        gen = DataGenerator(self.config)
        
        # Override config for this run
        gen.config['missingness']['mechanism'] = mechanism
        gen.config['missingness']['missing_rate'] = missing_rate
        
        # Generate Oracle
        data = gen.generate_oracle_data()
        data_miss = gen.introduce_missingness(data)
        
        X_full = data_miss['X'] # Scaled
        y_full = data_miss['y_oracle'] # True labels
        y_obs = data_miss['y_observed'] # With NaNs
        mask = data_miss['mask'] # 1=Obs, 0=Miss
        
        # 2. Split Train/Test
        # IMPORTANT: Missingness is usually on Training data (historical rejected).
        # We want to test on a Hold-Out set that is fully labeled (Oracle) to check performance recovery.
        # In real life, we only have observed data. But for simulation, we verify against P(Y=1|X) or Y_test.
        
        # Stratified split based on y_full to ensure balance
        X_train, X_test, y_train_full, y_test, mask_train, mask_test = train_test_split(
            X_full, y_full, mask, test_size=0.3, random_state=gen.seed, stratify=y_full
        )
        
        # Create y_train_obs matching the split
        y_train_obs = y_train_full.copy().astype(float)
        # Apply mask logic: locations where mask_train==0 should be NaN
        y_train_obs[mask_train == 0] = np.nan
        
        # 3. Correction / Pipeline
        start_time = time.time()
        
        model = xgb.XGBClassifier(eval_metric='logloss', random_state=gen.seed)
        # Or LogisticRegression for simpler interpretation
        # model = LogisticRegression(random_state=gen.seed)

        # Baseline: Oracle (Upper Bound) - Train on FULL labels
        if method == 'oracle':
            model.fit(X_train, y_train_full)
            
        elif method == 'complete_case':
            # Drop missing
            X_cc = X_train[mask_train == 1]
            y_cc = y_train_obs[mask_train == 1]
            model.fit(X_cc, y_cc)
            
        elif method == 'mean':
            # Impute Y with Mode (Most Frequent)
            # For binary classification, mean imputation of Y gives floats, 
            # which standard classifiers treat as classes if not handled strictly.
            # We use Mode here as a robust baseline.
            y_filled = pd.Series(y_train_obs).fillna(pd.Series(y_train_obs).mode()[0]).values
            model.fit(X_train, y_filled)
            
        elif method == 'zero_imputation':
             # Assume all rejected are Repaid (0)
            y_filled = y_train_obs.copy()
            y_filled[np.isnan(y_filled)] = 0
            model.fit(X_train, y_filled)
            
        elif method in ['mice', 'missforest']:
            # These are usually feature imputers.
            # Can we use them for Target Imputation?
            # Yes, if we include Y in the matrix [X, y].
            # Treat y as another column.
            
            # Combine X and y
            train_data = np.column_stack((X_train, y_train_obs))
            
            if method == 'mice': # Fixed key name from 'rows' to 'mice'
                imputer = MiceImputer(random_state=gen.seed)
            else:
                imputer = MissForestImputer(random_state=gen.seed)
                
            imputed_data = imputer.fit(train_data).transform(train_data)
            
            # Extract Y column
            y_imputed = imputed_data[:, -1]
            # Threshold to 0/1 to ensure binary labels
            y_imputed = (y_imputed > 0.5).astype(int)
            
            model.fit(X_train, y_imputed)
            
        elif method == 'reweighting':
            # IPW
            ipw = InverseProbabilityWeighter()
            ipw.fit(X_train, mask_train)
            weights = ipw.get_weights(X_train)
            
            # Train only on accepted, but with weights?
            # Standard IPW: Train on Observed, weigh by 1/P(Obs)
            X_cc = X_train[mask_train == 1]
            y_cc = y_train_obs[mask_train == 1]
            w_cc = weights[mask_train == 1]
            
            model.fit(X_cc, y_cc, sample_weight=w_cc)
            
        elif method == 'augmentation':
            aug = Augmentation(base_estimator=LogisticRegression()) # Base for labeling
            aug.fit(X_train, y_train_obs, mask_train)
            X_aug, y_aug = aug.transform(X_train, y_train_obs, mask_train)
            
            model.fit(X_aug, y_aug)
            
        else:
            raise ValueError(f"Unknown method {method}")
            
        # 4. Evaluate
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        metrics = self.evaluator.compute_metrics(y_test, y_pred_proba)
        
        metrics['method'] = method
        metrics['mechanism'] = mechanism
        metrics['missing_rate'] = missing_rate
        metrics['time'] = time.time() - start_time
        
        self.results.append(metrics)
        return metrics

    def run_all_experiments(self):
        """
        Main loop.
        """
        mechanisms = ['mcar', 'mnar_rejection'] # Add MAR if needed
        rates = [0.1, 0.3, 0.5]
        methods = [
            'oracle', 
            'complete_case', 
            'zero_imputation', 
            'mean',
            'mice',
            'missforest',
            'reweighting', 
            'augmentation'
        ]
        
        print(f"Starting experiments...")
        for mech in mechanisms:
            for rate in rates:
                print(f"Running {mech} @ {rate}")
                for method in methods:
                    try:
                        self.run_pipeline(mech, rate, method)
                    except Exception as e:
                        print(f"Failed {mech} {method}: {e}")
                        
        return pd.DataFrame(self.results)

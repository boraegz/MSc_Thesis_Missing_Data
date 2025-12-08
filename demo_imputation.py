from src.data.data_generator import DataGenerator
from src.imputation.baseline_imputer import BaselineImputer
from src.imputation.mice_imputer import MiceImputer
from src.utils import load_config
import numpy as np
import pandas as pd

# 1. Setup
print("--- STEP 1: GENERATE BROKEN DATA ---")
config = load_config("configs/experiment_config.yaml")
gen = DataGenerator(config)
data = gen.generate_oracle_data()
data_miss = gen.introduce_missingness(data)

# Combine X and y into one matrix for imputation demo
# (Usually we impute features, but let's simulate imputing the 'y' or features if they were missing)
# For this demo, let's pretend some FEATURES in X are missing too, or just use MICE on X.
# Wait, our thesis focus IS missing y (Reject Inference).
# MICE/MissForest use X_observed to predict Y_missing.

X = data_miss['X'] # Complete features
y_obs = data_miss['y_observed'] # Missing targets (NaNs)

# Combine for MICE
data_matrix = np.column_stack((X, y_obs))
print(f"Original Missing Values (NaNs): {np.isnan(data_matrix).sum()}")

# 2. Baseline Imputation
print("\n--- STEP 2: BASELINE IMPUTATION (Mean/Mode) ---")
# Use mode for binary target y
imputer_base = BaselineImputer(strategy='mode')
# We only impute the last column (y)
y_imputed_base = pd.Series(y_obs).fillna(pd.Series(y_obs).mode()[0]).values
print(f"NaNs after Baseline Imputation: {np.isnan(y_imputed_base).sum()}")
print(f"Value counts: {np.unique(y_imputed_base, return_counts=True)}")

# 3. MICE Imputation
print("\n--- STEP 3: MICE IMPUTATION (Smart) ---")
imputer_mice = MiceImputer(max_iter=5)
data_imputed_mice = imputer_mice.fit(data_matrix).transform(data_matrix)
print(f"NaNs after MICE Imputation: {np.isnan(data_imputed_mice).sum()}")

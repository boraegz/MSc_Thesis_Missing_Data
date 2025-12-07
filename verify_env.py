import sys
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels.api as sm
import torch
import missingno
import fancyimpute
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
import lightgbm
import joblib

print(f"Python version: {sys.version}")
print(f"Numpy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"XGBoost version: {xgboost.__version__}")
print(f"LightGBM version: {lightgbm.__version__}")

print("\nSUCCESS: All critical libraries imported successfully!")

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
import numpy as np
import pandas as pd
from scipy.stats import norm

class MissingDataHandler:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='mean')

    def impute_missing_data(self, data):
        data_imputed = data.copy()
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data_imputed[numeric_columns] = self.imputer.fit_transform(data[numeric_columns])
        return data_imputed

    def heckman_correction(self, data):
        missing_mask = data.isnull().any(axis=1)
        X = data.fillna(data.mean())
        probit_model = LogisticRegression()
        probit_model.fit(X, missing_mask)
        z_scores = probit_model.predict_proba(X)[:, 1]
        inverse_mills = norm.pdf(z_scores) / (1 - norm.cdf(z_scores))
        data_with_mills = data.copy()
        data_with_mills['inverse_mills'] = inverse_mills
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].isnull().any():
                mask = ~data[col].isnull()
                X_obs = data_with_mills.loc[mask].drop(columns=[col, 'inverse_mills'])
                y_obs = data_with_mills.loc[mask, col]
                X_obs['inverse_mills'] = inverse_mills[mask]
                correction_model = LinearRegression()
                correction_model.fit(X_obs, y_obs)
                missing_idx = data[col].isnull()
                X_miss = data_with_mills.loc[missing_idx].drop(columns=[col, 'inverse_mills'])
                X_miss['inverse_mills'] = inverse_mills[missing_idx]
                data.loc[missing_idx, col] = correction_model.predict(X_miss)
        return data

    def basl_correction(self, data):
        missing_mask = data.isnull()
        X = data.fillna(data.mean())
        corrected_data = data.copy()
        for col in data.columns:
            if data[col].isnull().any():
                X_col = X.drop(columns=[col])
                y_col = missing_mask[col].astype(int)
                basl_model = LogisticRegression()
                basl_model.fit(X_col, y_col)
                missing_probs = basl_model.predict_proba(X_col)[:, 1]
                missing_idx = data[col].isnull()
                if data[col].dtype in ['int64', 'float64']:
                    observed_values = data[col].dropna()
                    corrected_data.loc[missing_idx, col] = np.random.choice(
                        observed_values,
                        size=missing_idx.sum(),
                        p=missing_probs[missing_idx]/missing_probs[missing_idx].sum()
                    )
                else:
                    value_counts = data[col].value_counts(normalize=True)
                    corrected_data.loc[missing_idx, col] = np.random.choice(
                        value_counts.index,
                        size=missing_idx.sum(),
                        p=value_counts * missing_probs[missing_idx].reshape(-1,1)
                    )
        return corrected_data

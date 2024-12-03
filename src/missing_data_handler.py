from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from statsmodels.imputation import mice
from sklearn.linear_model import LogisticRegression

class MissingDataHandler:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='mean')
        self.basl_model = None

    def impute_missing_data(self, data):
        """
        Impute missing values using mean strategy.
        Args:
            data (pd.DataFrame): Data to impute.
        Returns:
            pd.DataFrame: Data with imputed missing values.
        """
        data_imputed = data.copy()
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        data_imputed[numeric_columns] = self.imputer.fit_transform(data[numeric_columns])
        return data_imputed

    def heckman_correction(self, data):
        """
        Apply Heckman correction for MNAR.
        Args:
            data (pd.DataFrame): Data to correct.
        Returns:
            pd.DataFrame: Corrected data using Heckman.
        """
        # Heckman correction example (simplified using MICE)
        imp = mice.MICEData(data)
        data_imputed = imp.fit()
        return data_imputed.data

    def basl_correction(self, data):
        """
        Apply BASL method to handle bias in MNAR.
        Args:
            data (pd.DataFrame): Data to correct.
        Returns:
            pd.DataFrame: Corrected data using BASL.
        """
        X = data.drop(columns='Repayment_Label')
        y = data['Repayment_Label'].isnull().astype(int)
        self.basl_model = LogisticRegression()
        self.basl_model.fit(X, y)
        missing_pred = self.basl_model.predict_proba(X)[:, 1]
        data['Repayment_Label'] = data.apply(
            lambda row: row['Repayment_Label'] if not np.isnan(row['Repayment_Label']) 
            else np.random.choice([0, 1], p=[1-missing_pred[row.name], missing_pred[row.name]]), axis=1)
        return data

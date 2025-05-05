# custom_transformers.py

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats.mstats import winsorize

class InitialCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns_to_drop = ['nameOrig', 'nameDest']
        self.null_like_values = ['', ' ', 'nan', 'NaN', 'NULL', 'None']
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X.drop(columns=self.columns_to_drop, errors='ignore', inplace=True)
        X.replace(self.null_like_values, np.nan, inplace=True)
        return X

class Winsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, limits=(0.01, 0.01)):
        self.limits = limits

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X)
        X_winsorized = np.apply_along_axis(
            lambda col: winsorize(col, limits=self.limits), axis=0, arr=X
        )
        return X_winsorized

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class IQRCapper(BaseEstimator, TransformerMixin):
    """Winsorise each feature at the IQR fence learned from the fitted (training) data only."""
    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.percentile(X, 25, axis=0), np.percentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

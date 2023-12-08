from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
import numpy as np


class FeatureQuantileTransformer(QuantileTransformer):
    def __init__(self, n_quantiles=1000, output_distribution='normal', **kwargs):
        super().__init__(n_quantiles=n_quantiles, output_distribution=output_distribution, **kwargs)


class ClippingStandardScaler(TransformerMixin):
    def __init__(self, clip_range=(-4, 4)):
        self.scaler = StandardScaler()
        self.clip_range = clip_range

    def fit(self, X, y=None):
        self.scaler.fit(X)
        return self

    def transform(self, X, y=None):
        scaled_data = self.scaler.transform(X)
        clipped_data = np.clip(scaled_data, *self.clip_range)
        return clipped_data

    def inverse_transform(self, X, copy=None):
        return self.scaler.inverse_transform(X, copy)


class LogAndQuantileScaler(TransformerMixin):
    def __init__(self):
#         self.log_transformer = np.log
        self.scaler = FeatureQuantileTransformer()

    def fit(self, X, y=None):
#         log_transformed = self.log_transformer(X + 1e-8)
        self.scaler.fit(X)
        return self

    def transform(self, X, y=None):
#         log_transformed = self.log_transformer(X)
        standardized_values = self.scaler.transform(X.reshape(-1, 1))
        return standardized_values.flatten()


class LogAndStandardScaler(TransformerMixin):
    def __init__(self):
        self.log_transformer = np.log
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        log_transformed = self.log_transformer(X)
        self.scaler.fit(log_transformed)
        return self

    def transform(self, X, y=None):
        log_transformed = self.log_transformer(X)
        standardized_values = self.scaler.transform(log_transformed.reshape(-1, 1))
        return standardized_values.flatten()

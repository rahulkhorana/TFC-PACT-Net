import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics.pairwise import pairwise_kernels


class TanimotoGP:
    def __init__(self):
        self.X_train = None
        self.gp = GaussianProcessRegressor(
            kernel="precomputed", alpha=1e-6, normalize_y=True  # type: ignore
        )

    def fit(self, X, y):
        self.X_train = X
        K = pairwise_kernels(X, X, metric="tanimoto")
        self.gp.fit(K, y)

    def predict(self, X):
        K_test = pairwise_kernels(X, self.X_train, metric="tanimoto")
        return self.gp.predict(K_test, return_std=True)

    def score(self, X, y):
        K_test = pairwise_kernels(X, self.X_train, metric="tanimoto")
        return self.gp.score(K_test, y)

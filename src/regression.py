import numpy as np
from fastcore.utils import *


class Regression:

    def __init__(self, n_iter, lr, regularization=None):
        store_attr()

    def _init_weight(self, n_features):
        w = np.random.normal(loc=0, scale=1, size=n_features)
        return w

    def fit(self, x, y):
        x = (x - np.mean(x, axis=0)) / (np.std(x, axis=0))
        x = np.c_[np.ones(x.shape[0]), x]
        self.w = self._init_weight(x.shape[1])
        for _ in range(self.n_iter):
            y_pred = x @ self.w
            if self.regularization is not None:
                mse = np.mean(0.5 * (y - y_pred) ** 2 + self.reguralization(self.w))
            else:
                mse = np.mean(0.5 * (y - y_pred) ** 2)
                grad_w = -(y - y_pred) @ x
                self.w -= self.lr * grad_w

    def predict(self, x):
        x = np.c_[np.ones(x.shape[0]), x]
        y_pred = x @ self.w
        return y_pred


class LassoRegularization:
    def __init__(self, alpha):
        store_attr()

    def __call__(self, w):
        return self.alpha * np.linalg.norm(w)

    def grad(self, w):
        return self.alpha * np.sign(w)


class RidgeRegularization:

    def __init__(self, alpha):
        store_attr()

    def __call__(self, w):
        return 0.5 * self.alpha * w @ w

    def grad(self, w):
        return self.alpha * w


class ElasticRegression:
    def __init__(self, alpha, contribution=0.5):
        store_attr()

    def __call__(self, w):
        lasso = self.contribution * np.linalg.norm(w)
        ridge = (1 - self.contribution) * w @ w
        return self.alpha * (lasso + ridge)

    def grad(self, w):
        lasso = self.contribution * np.sign(w)
        ridge = (1 - self.contribution) * w
        return self.alpha * (lasso + ridge)



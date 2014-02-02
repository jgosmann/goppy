"""Module providing an online Gaussian process."""

import numpy as np
from numpy.linalg import cholesky, inv


# TODO unit test this class
class SquaredExponentialKernel(object):
    def __init__(self, lengthscales, variance=1.0):
        self.lengthscales = np.asarray(lengthscales)
        self.variance = variance

    def __call__(self, x1, x2):
        d = self._calc_distance(x1, x2)
        return self.variance * np.exp(-0.5 * d / self.lengthscales ** 2)

    def _calc_distance(self, x1, x2):
        return -2 * np.dot(x1, x2.T) + (
            np.sum(np.square(x1), 1)[:, None] +
            np.sum(np.square(x2), 1)[None, :])


class OnlineGP(object):
    def __init__(self, kernel, noise_var=0.0):
        self.kernel = kernel
        self.noise_var = noise_var
        self.x_train = None
        self.y_train = None
        self.L_inv = None
        self.K_inv = None

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y
        # FIXME store only L_inv or K_inv
        self.L_inv = inv(cholesky(
            self.kernel(x, x) + np.eye(len(x)) * self.noise_var))
        self.K_inv = np.dot(self.L_inv.T, self.L_inv)

    def predict(self, x):
        K_new_vs_old = self.kernel(x, self.x_train)

        svs = np.dot(self.K_inv, self.y_train)
        return np.dot(K_new_vs_old, svs)

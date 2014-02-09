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

    def diag(self, x1, x2):
        if x1 is x2:
            return self.variance * np.ones(len(x1))

        d = -2 * np.einsum('ij,ij->i', x1, x2) + (
            np.sum(np.square(x1), 1) + np.sum(np.square(x2), 1))
        return self.variance * np.exp(-0.5 * d / self.lengthscale ** 2)

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
        self.inv_chol = None
        self.__inv_cov_matrix = None

    def _get_inv_cov_matrix(self):
        if self.__inv_cov_matrix is None:
            self.__inv_cov_matrix = np.dot(self.inv_chol.T, self.inv_chol)
        return self.__inv_cov_matrix

    def _del_inv_cov_matrix(self):
        self.__inv_cov_matrix = None

    inv_cov_matrix = property(_get_inv_cov_matrix, fdel=_del_inv_cov_matrix)

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y
        self.inv_chol = inv(cholesky(
            self.kernel(x, x) + np.eye(len(x)) * self.noise_var))
        del self.inv_cov_matrix

    def predict(self, x, what=['mean']):
        pred = {}

        input_vs_train_dist = self.kernel(x, self.x_train)

        if 'mean' in what:
            pred['mean'] = self._calc_mean_prediction(input_vs_train_dist)
        if 'mse' in what:
            pred['mse'] = self._calc_mse_prediction(x, input_vs_train_dist)

        return pred

    def _calc_mean_prediction(self, input_vs_train_dist):
        svs = np.dot(self.inv_cov_matrix, self.y_train)
        return np.dot(input_vs_train_dist, svs)

    def _calc_mse_prediction(self, x, input_vs_train_dist):
        svs = np.dot(self.inv_cov_matrix, input_vs_train_dist.T)
        return np.maximum(
            self.noise_var,
            self.noise_var + self.kernel.diag(x, x) - np.einsum(
                'ij,ji->i', input_vs_train_dist, svs))

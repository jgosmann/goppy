"""Module providing an online Gaussian process."""

import numpy as np
from numpy.linalg import cholesky, inv

from .growable import GrowableArray


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
        self.trained = False

    def _get_inv_cov_matrix(self):
        if self.__inv_cov_matrix is None:
            self.__inv_cov_matrix = np.dot(self.inv_chol.T, self.inv_chol)
        return self.__inv_cov_matrix

    def _del_inv_cov_matrix(self):
        self.__inv_cov_matrix = None

    inv_cov_matrix = property(_get_inv_cov_matrix, fdel=_del_inv_cov_matrix)

    def fit(self, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        # FIXME expected shape
        self.x_train = GrowableArray(x.shape)
        self.y_train = GrowableArray(y.shape)
        self.x_train[:, :] = x
        self.y_train[:, :] = y
        self.inv_chol = GrowableArray((x.shape[0], x.shape[0]))
        self.inv_chol[:, :] = inv(cholesky(
            self.kernel(x, x) + np.eye(len(x)) * self.noise_var))
        del self.inv_cov_matrix
        self.trained = True

    def add(self, x, y):
        # TODO trained unit test
        if len(x) <= 0:
            return

        x = np.asarray(x)
        y = np.asarray(y)

        if not self.trained:
            self.fit(x, y)
            return

        input_vs_train_dist = self.kernel(x, self.x_train)
        proj = np.dot(input_vs_train_dist, self.inv_chol.T)
        covmat = self.kernel(x, x) + np.eye(len(x)) * self.noise_var - \
            np.dot(proj, proj.T)
        diag_indices = np.diag_indices_from(covmat)
        covmat[diag_indices] = np.maximum(self.noise_var, covmat[diag_indices])

        self.x_train.grow_by((len(x), 0))
        self.y_train.grow_by((len(y), 0))
        self.x_train[-len(x):, :] = x
        self.y_train[-len(y):, :] = y

        #try: TODO
        new_inv_chol = inv(cholesky(covmat))
        #except linalg.LinAlgError:
            #warnings.warn(
                #'New submatrix of covariance matrix singular. '
                #'Retraining on all data.', NumericalStabilityWarning)
            #self._refit()
            #return

        l = len(self.inv_chol)
        self.inv_chol.grow_by((len(x), len(x)))
        self.inv_chol[:l, l:] = 0.0
        self.inv_chol[l:, :l] = -np.dot(
            np.dot(new_inv_chol, proj), self.inv_chol[:l, :l])
        self.inv_chol[l:, l:] = new_inv_chol
        del self.inv_cov_matrix

    def predict(self, x, what=('mean',)):
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

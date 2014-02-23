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
        return self.full(x1, x2, what=('y',))['y']

    def full(self, x1, x2, what=('y',)):
        res = {}
        d = self._calc_distance(x1, x2)
        res['y'] = self.variance * np.exp(-0.5 * d / self.lengthscales ** 2)
        if 'derivative' in what:
            direction = x1[:, None, :] - x2[None, :, :]
            res['derivative'] = (-1.0 / (self.lengthscales ** 2) * direction *
                res['y'][:, :, None])
        return res

    def diag(self, x1, x2):
        if x1 is x2:
            return self.variance * np.ones(len(x1))

        d = -2 * np.einsum('ij,ij->i', x1, x2) + (
            np.sum(np.square(x1), 1) + np.sum(np.square(x2), 1))
        return self.variance * np.exp(-0.5 * d / self.lengthscales ** 2)

    @staticmethod
    def _calc_distance(x1, x2):
        return -2 * np.dot(x1, x2.T) + (
            np.sum(np.square(x1), 1)[:, None] +
            np.sum(np.square(x2), 1)[None, :])


class OnlineGP(object):
    def __init__(
            self, kernel, noise_var=0.0, expected_size=None,
            buffer_factory=GrowableArray):
        self.kernel = kernel
        self.noise_var = noise_var
        self._expected_size = expected_size
        self._buffer_factory = buffer_factory
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

        if self._expected_size is not None:
            buffer_shape = (self._expected_size,)
            buffer_shape2 = (self._expected_size, self._expected_size)
        else:
            buffer_shape = buffer_shape2 = None

        self.x_train = self._buffer_factory(x.shape, buffer_shape=buffer_shape)
        self.y_train = self._buffer_factory(y.shape, buffer_shape=buffer_shape)
        self.x_train[:, :] = x
        self.y_train[:, :] = y
        self.inv_chol = self._buffer_factory(
            (x.shape[0], x.shape[0]), buffer_shape=buffer_shape2)
        self.inv_chol[:, :] = inv(cholesky(
            self.kernel(x, x) + np.eye(len(x)) * self.noise_var))
        del self.inv_cov_matrix
        self.trained = True

    def add(self, x, y):
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

        new_inv_chol = inv(cholesky(covmat))

        l = len(self.inv_chol)
        self.inv_chol.grow_by((len(x), len(x)))
        self.inv_chol[:l, l:] = 0.0
        self.inv_chol[l:, :l] = -np.dot(
            np.dot(new_inv_chol, proj), self.inv_chol[:l, :l])
        self.inv_chol[l:, l:] = new_inv_chol
        del self.inv_cov_matrix

    def predict(self, x, what=('mean',)):
        pred = {}

        if 'derivative' in what or 'mse_derivative' in what:
            kernel_what = ('y', 'derivative')
        else:
            kernel_what = ('y',)

        lazy_vars = LazyVarCollection(
            input_vs_train_dist=lambda v: self.kernel.full(
                x, self.x_train, kernel_what),
            svs=lambda v: np.dot(self.inv_cov_matrix, self.y_train),
            mean=lambda v: np.dot(v.input_vs_train_dist['y'], v.svs),
            mse_svs=lambda v: np.dot(
                self.inv_cov_matrix, v.input_vs_train_dist['y'].T),
            mse=lambda v: np.maximum(
                self.noise_var,
                self.noise_var + self.kernel.diag(x, x) - np.einsum(
                    'ij,ji->i', v.input_vs_train_dist['y'], v.mse_svs)),
            derivative=lambda v: np.einsum(
                'ijk,jl->ilk', v.input_vs_train_dist['derivative'], v.svs),
            mse_derivative=lambda v: -2 * np.einsum(
                'ijk,ji->ik', v.input_vs_train_dist['derivative'], v.mse_svs))

        if 'mean' in what:
            pred['mean'] = lazy_vars.mean
        if 'mse' in what:
            pred['mse'] = lazy_vars.mse
        if 'derivative' in what:
            pred['derivative'] = lazy_vars.derivative
        if 'mse_derivative' in what:
            pred['mse_derivative'] = lazy_vars.mse_derivative
        return pred

    def calc_log_likelihood(self):
        svs = np.dot(self.inv_chol, self.y_train)
        log_likelihood = -0.5 * np.dot(svs.T, svs) + \
            np.sum(np.log(np.diag(self.inv_chol))) - \
            0.5 * len(self.y_train) * np.log(2 * np.pi)
        return log_likelihood


# TODO unit test
class LazyVarCollection(object):
    def __init__(self, **kwargs):
        self._eval_fns = kwargs

    def __getattr__(self, name):
        value = self._eval_fns[name](self)
        setattr(self, name, value)
        return value

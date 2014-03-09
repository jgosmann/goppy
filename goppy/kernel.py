"""Module providing kernels for the use with Gaussian processes."""

import numpy as np


class ExponentialKernel(object):
    def __init__(self, lengthscales, variance=1.0):
        self.lengthscales = np.asarray(lengthscales)
        self.variance = variance

    @property
    def params(self):
        return np.concatenate((self.lengthscales, (self.variance,)))

    @params.setter
    def params(self, values):
        self.lengthscales = np.asarray(values[:-1])
        self.variance = values[-1]

    def __call__(self, x1, x2):
        return self.full(x1, x2, what=('y',))['y']

    def full(self, x1, x2, what=('y',)):
        res = {}
        d = self._calc_distance(x1, x2)
        res['y'] = self.variance * np.exp(-d / self.lengthscales)
        if 'derivative' in what:
            direction = x1[:, None, :] - x2[None, :, :]
            res['derivative'] = -direction / d[:, :, None] / \
                self.lengthscales * res['y'][:, :, None]
        if 'param_derivatives' in what:
            variance_deriv = np.exp(-d / self.lengthscales)
            lengthscale_deriv = self.variance * d / (self.lengthscales ** 2) *\
                variance_deriv
            res['param_derivatives'] = [lengthscale_deriv, variance_deriv]
        return res

    def diag(self, x1, x2):
        if x1 is x2:
            return self.variance * np.ones(len(x1))

        d = np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
        return self.variance * np.exp(-d / self.lengthscales)

    @staticmethod
    def _calc_distance(x1, x2):
        return np.sqrt(np.maximum(0, -2 * np.dot(x1, x2.T) + (
            np.sum(np.square(x1), 1)[:, None] +
            np.sum(np.square(x2), 1)[None, :])))


class SquaredExponentialKernel(object):
    def __init__(self, lengthscales, variance=1.0):
        self.lengthscales = np.asarray(lengthscales)
        self.variance = variance

    def _get_params(self):
        return np.concatenate((self.lengthscales, (self.variance,)))

    def _set_params(self, value):
        self.lengthscales = np.asarray(value[:-1])
        self.variance = value[-1]

    params = property(_get_params, _set_params)

    def __call__(self, x1, x2):
        return self.full(x1, x2, what=('y',))['y']

    def full(self, x1, x2, what=('y',)):
        res = {}
        d = self._calc_distance(x1, x2)
        res['y'] = self.variance * np.exp(-0.5 * d / self.lengthscales ** 2)
        if 'derivative' in what:
            direction = x1[:, None, :] - x2[None, :, :]
            res['derivative'] = (
                -1.0 / (self.lengthscales ** 2) * direction *
                res['y'][:, :, None])
        if 'param_derivatives' in what:
            variance_deriv = np.exp(-0.5 * d / self.lengthscales ** 2)
            lengthscale_deriv = (
                2 * self.variance * d / (self.lengthscales ** 3) *
                variance_deriv)
            res['param_derivatives'] = [lengthscale_deriv, variance_deriv]
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

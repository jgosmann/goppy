"""Provides kernels for the use with Gaussian processes."""

import numpy as np


class Kernel(object):
    """Abstract base class for kernels."""

    def full(self, x1, x2, what=('y',)):
        r"""Evaluate the kernel for all pairs of `x1` and `x2`.

        Depending on the values included in the `what` parameter different
        evaluations will be made:

        * ``'y'``: Evaluate the kernel for each pair of `x1` and `x2` resulting
          in the Gram matrix.
        * ``'derivative'``: Evaluate the partial derivatives.
          ``res['derivative'][i, j, :]`` will correspond to
          :math:`\left(\frac{\partial k}{d\mathtt{x2}}\right)
          \left(\mathtt{x1}_i, \mathtt{x2}_j\right)` with subscripts denoting
          input data points, and the kernel
          :math:`k(\mathtt{x1}, \mathtt{x2})`.
        * ``'param_derivatives'``: Evaluate the partial derivatives of the
          kernel parameters. ``res['param_derivatives']`` will be a list with
          the :math:`i`-th element corresponding to
          :math:`\left(\frac{\partial k}{d\theta_i}\right)
          \left(\mathtt{x1}, \mathtt{x2}\right)` wherein :math:`\theta_i` is
          the :math:`i`-th parameter. The order of the parameters is the same
          as in attr:`params`.

        An implementation of a kernel is not required to provide the
        functionality to evaluate ``'derivative'`` and/or
        ``'param_derivatives'``. In this case the set of available predictions
        of a Gaussian Process might be limited. All the GopPy standard kernels
        implement the complete functionality described above.

        Parameters
        ----------
        x1, x2 : (`N`, `D`) array-like
            The `N` data points of dimension `D` to evaluate the kernel for.
        what : set-like, optional
            Types of evaluations to be made (see above).

        Returns
        -------
        dict
            Dictionary with the elements of `what` as keys and the
            corresponding evaluations as values.

        """
        raise NotImplementedError()


class ExponentialKernel(Kernel):
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


class Matern32Kernel(Kernel):
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
        scaled_d = np.sqrt(3) * self._calc_distance(x1, x2) / self.lengthscales
        exp_term = np.exp(-scaled_d)
        res['y'] = self.variance * (1 + scaled_d) * exp_term
        if 'derivative' in what:
            direction = x1[:, None, :] - x2[None, :, :]
            res['derivative'] = -3 * direction / (self.lengthscales ** 2) *\
                self.variance * exp_term[:, :, None]
        if 'param_derivatives' in what:
            variance_deriv = (1 + scaled_d) * exp_term
            lengthscale_deriv = self.variance / self.lengthscales * \
                (scaled_d ** 2) * exp_term
            res['param_derivatives'] = [lengthscale_deriv, variance_deriv]
        return res

    def diag(self, x1, x2):
        if x1 is x2:
            return self.variance * np.ones(len(x1))

        d = np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
        scaled_d = np.sqrt(3) * d / self.lengthscales
        return self.variance * (1 + scaled_d) * np.exp(-scaled_d)

    @staticmethod
    def _calc_distance(x1, x2):
        return np.sqrt(np.maximum(0, -2 * np.dot(x1, x2.T) + (
            np.sum(np.square(x1), 1)[:, None] +
            np.sum(np.square(x2), 1)[None, :])))


class Matern52Kernel(Kernel):
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
        scaled_d = np.sqrt(5) * d / self.lengthscales
        exp_term = np.exp(-scaled_d)
        res['y'] = self.variance * (1 + scaled_d + scaled_d ** 2 / 3.0) *\
            exp_term
        if 'derivative' in what:
            direction = x1[:, None, :] - x2[None, :, :]
            res['derivative'] = -5.0 / 3.0 * direction /\
                (self.lengthscales ** 2) * self.variance *\
                ((d + np.sqrt(5) * d ** 2 / self.lengthscales) * exp_term /
                    d)[:, :, None]
        if 'param_derivatives' in what:
            der_d = np.sqrt(3) * d / self.lengthscales
            der_exp_term = np.exp(-der_d)
            variance_deriv = (1 + der_d + der_d ** 2 / 3.0) * der_exp_term
            lengthscale_deriv = self.variance / self.lengthscales * \
                (der_d ** 3 / 3.0 + der_d ** 2) * der_exp_term
            res['param_derivatives'] = [lengthscale_deriv, variance_deriv]
        return res

    def diag(self, x1, x2):
        if x1 is x2:
            return self.variance * np.ones(len(x1))

        d = np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
        scaled_d = np.sqrt(5) * d / self.lengthscales
        return self.variance * (1 + scaled_d + scaled_d ** 2 / 3.0) * \
            np.exp(-scaled_d)

    @staticmethod
    def _calc_distance(x1, x2):
        return np.sqrt(np.maximum(0, -2 * np.dot(x1, x2.T) + (
            np.sum(np.square(x1), 1)[:, None] +
            np.sum(np.square(x2), 1)[None, :])))


class SquaredExponentialKernel(Kernel):
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

# -*- coding: utf-8 -*-
"""Provides kernels for the use with Gaussian processes."""

import numpy as np


class Kernel:
    """Abstract base class for kernels.

    An instance of this class is callable and ``instance(x1, x2)`` will call
    ``instance.full(x1, x2)``.

    Attributes
    ----------
    params : 1d ndarray
        Array representation of the kernel parameters.
    """

    def __call__(self, x1, x2):
        return self.full(x1, x2, what=("y",))["y"]

    def full(self, x1, x2, what=("y",)):
        r"""Evaluate the kernel for all pairs of `x1` and `x2`.

        Depending on the values included in the `what` parameter different
        evaluations will be made and returned as a dictionary ``res``:

        * ``'y'``: Evaluate the kernel for each pair of `x1` and `x2` resulting
          in the Gram matrix.
        * ``'derivative'``: Evaluate the partial derivatives.
          ``res['derivative'][i, j, :]`` will correspond to
          :math:`\left(\frac{\partial k}{d\mathtt{x2}}\right)
          \left(\mathtt{x1}_i, \mathtt{x2}_j\right)` with subscripts denoting
          input data points, and the kernel
          :math:`k(\mathtt{x1}, \mathtt{x2})`.
        * ``'param_derivatives'``: Evaluate the partial derivatives of the
          kernel parameters. ``res['param_derivatives']`` will be a
          list with the :math:`i`-th element corresponding to
          :math:`\left(\frac{\partial k}{d\theta_i}\right)
          \left(\mathtt{x1}, \mathtt{x2}\right)` wherein :math:`\theta_i` is
          the :math:`i`-th parameter. The order of the parameters is the same
          as in the :attr:`params` attribute.

        An implementation of a kernel is not required to provide the
        functionality to evaluate ``'derivative'`` and/or
        ``'param_derivatives'``. In this case the set of available predictions
        of a Gaussian process might be limited. All the GopPy standard kernels
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

        See Also
        --------
        diag

        """
        raise NotImplementedError()

    def diag(self, x1, x2):
        """Evaluate the kernel and return only the diagonal of the Gram matrix.

        If only the diagonal is needed, this functions may be more efficient
        than calculating the full Gram matrix with :func:`full`.

        Parameters
        ----------
        x1, x2 : (`N`, `D`) array-like
            The `N` data points of dimension `D` to evaluate the kernel for.

        Returns
        -------
        1d ndarray
            The diagonal of the resulting Gram matrix from evaluating the
            kernels for pairs from `x1` and `x2`.

        See Also
        --------
        full

        """
        return np.diag(self(x1, x2))


class ExponentialKernel(Kernel):
    r"""Exponential kernel.

    The exponential kernel is defined as :math:`k(r_i) = \sigma^2
    \exp\left(-\frac{r_i}{l_i}\right)` with
    :math:`r = |\mathtt{x1} - \mathtt{x2}|`, kernel variance :math:`\sigma^2`
    and length scales :math:`l`.

    Parameters
    ----------
    lengthscales : (`D`,) array-like
        The length scale :math:`l_i` for each dimension.
    variance : float
        The kernel variance :math:`\sigma^2`.
    """

    def __init__(self, lengthscales, variance=1.0):
        self.lengthscales = np.asarray(lengthscales)
        self.variance = variance

    @property
    def params(self):
        """1d-array of kernel parameters.

        The first `D` values are the length scales for each dimension and the
        last value is the kernel variance.
        """
        return np.concatenate((self.lengthscales, (self.variance,)))

    @params.setter
    def params(self, values):
        self.lengthscales = np.asarray(values[:-1])
        self.variance = values[-1]

    def full(self, x1, x2, what=("y",)):
        res = {}
        d = self._calc_distance(x1, x2)
        res["y"] = self.variance * np.exp(-d / self.lengthscales)
        if "derivative" in what:
            direction = x1[:, None, :] - x2[None, :, :]
            res["derivative"] = (
                -direction / d[:, :, None] / self.lengthscales * res["y"][:, :, None]
            )
        if "param_derivatives" in what:
            variance_deriv = np.exp(-d / self.lengthscales)
            lengthscale_deriv = (
                self.variance * d / (self.lengthscales**2) * variance_deriv
            )
            res["param_derivatives"] = [lengthscale_deriv, variance_deriv]
        return res

    def diag(self, x1, x2):
        if x1 is x2:
            return self.variance * np.ones(len(x1))

        d = np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
        return self.variance * np.exp(-d / self.lengthscales)

    @staticmethod
    def _calc_distance(x1, x2):
        return np.sqrt(
            np.maximum(
                0,
                -2 * np.dot(x1, x2.T)
                + (
                    np.sum(np.square(x1), 1)[:, None]
                    + np.sum(np.square(x2), 1)[None, :]
                ),
            )
        )


class Matern32Kernel(Kernel):
    r"""Matérn 3/2 kernel.

    The Matérn kernel with :math:`\nu = \frac{3}{2}` is defined as
    :math:`k(r_i) = \sigma^2 \left(1 + \frac{r_i \sqrt{3}}{l_i}\right)
    \exp\left(-\frac{r_i \sqrt{3}}{l_i}\right)` with
    :math:`r = |\mathtt{x1} - \mathtt{x2}|`, kernel variance :math:`\sigma^2`
    and length scales :math:`l`.

    Parameters
    ----------
    lengthscales : (`D`,) array-like
        The length scale :math:`l_i` for each dimension.
    variance : float
        The kernel variance :math:`\sigma^2`.
    """

    def __init__(self, lengthscales, variance=1.0):
        self.lengthscales = np.asarray(lengthscales)
        self.variance = variance

    @property
    def params(self):
        """1d-array of kernel parameters.

        The first `D` values are the length scales for each dimension and the
        last value is the kernel variance.
        """
        return np.concatenate((self.lengthscales, (self.variance,)))

    @params.setter
    def params(self, values):
        self.lengthscales = np.asarray(values[:-1])
        self.variance = values[-1]

    def full(self, x1, x2, what=("y",)):
        res = {}
        scaled_d = np.sqrt(3) * self._calc_distance(x1, x2) / self.lengthscales
        exp_term = np.exp(-scaled_d)
        res["y"] = self.variance * (1 + scaled_d) * exp_term
        if "derivative" in what:
            direction = x1[:, None, :] - x2[None, :, :]
            res["derivative"] = (
                -3
                * direction
                / (self.lengthscales**2)
                * self.variance
                * exp_term[:, :, None]
            )
        if "param_derivatives" in what:
            variance_deriv = (1 + scaled_d) * exp_term
            lengthscale_deriv = (
                self.variance / self.lengthscales * (scaled_d**2) * exp_term
            )
            res["param_derivatives"] = [lengthscale_deriv, variance_deriv]
        return res

    def diag(self, x1, x2):
        if x1 is x2:
            return self.variance * np.ones(len(x1))

        d = np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
        scaled_d = np.sqrt(3) * d / self.lengthscales
        return self.variance * (1 + scaled_d) * np.exp(-scaled_d)

    @staticmethod
    def _calc_distance(x1, x2):
        return np.sqrt(
            np.maximum(
                0,
                -2 * np.dot(x1, x2.T)
                + (
                    np.sum(np.square(x1), 1)[:, None]
                    + np.sum(np.square(x2), 1)[None, :]
                ),
            )
        )


class Matern52Kernel(Kernel):
    r"""Matérn 5/2 kernel.

    The Matérn kernel with :math:`\nu = \frac{5}{2}` is defined as
    :math:`k(r_i) = \sigma^2 \left(1 + \frac{r_i \sqrt{5}}{l_i} +
    \frac{5 r_i^2}{3 l_i^2}\right)
    \exp\left(-\frac{r_i \sqrt{5}}{l_i}\right)` with
    :math:`r = |\mathtt{x1} - \mathtt{x2}|`, kernel variance :math:`\sigma^2`
    and length scales :math:`l`.

    Parameters
    ----------
    lengthscales : (`D`,) array-like
        The length scale :math:`l_i` for each dimension.
    variance : float
        The kernel variance :math:`\sigma^2`.
    """

    def __init__(self, lengthscales, variance=1.0):
        self.lengthscales = np.asarray(lengthscales)
        self.variance = variance

    @property
    def params(self):
        """1d-array of kernel parameters.

        The first `D` values are the length scales for each dimension and the
        last value is the kernel variance.
        """
        return np.concatenate((self.lengthscales, (self.variance,)))

    @params.setter
    def params(self, values):
        self.lengthscales = np.asarray(values[:-1])
        self.variance = values[-1]

    def full(self, x1, x2, what=("y",)):
        res = {}
        d = self._calc_distance(x1, x2)
        scaled_d = np.sqrt(5) * d / self.lengthscales
        exp_term = np.exp(-scaled_d)
        res["y"] = self.variance * (1 + scaled_d + scaled_d**2 / 3.0) * exp_term
        if "derivative" in what:
            direction = x1[:, None, :] - x2[None, :, :]
            res["derivative"] = (
                -5.0
                / 3.0
                * direction
                / (self.lengthscales**2)
                * self.variance
                * ((d + np.sqrt(5) * d**2 / self.lengthscales) * exp_term / d)[
                    :, :, None
                ]
            )
        if "param_derivatives" in what:
            der_d = np.sqrt(3) * d / self.lengthscales
            der_exp_term = np.exp(-der_d)
            variance_deriv = (1 + der_d + der_d**2 / 3.0) * der_exp_term
            lengthscale_deriv = (
                self.variance
                / self.lengthscales
                * (der_d**3 / 3.0 + der_d**2)
                * der_exp_term
            )
            res["param_derivatives"] = [lengthscale_deriv, variance_deriv]
        return res

    def diag(self, x1, x2):
        if x1 is x2:
            return self.variance * np.ones(len(x1))

        d = np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
        scaled_d = np.sqrt(5) * d / self.lengthscales
        return self.variance * (1 + scaled_d + scaled_d**2 / 3.0) * np.exp(-scaled_d)

    @staticmethod
    def _calc_distance(x1, x2):
        return np.sqrt(
            np.maximum(
                0,
                -2 * np.dot(x1, x2.T)
                + (
                    np.sum(np.square(x1), 1)[:, None]
                    + np.sum(np.square(x2), 1)[None, :]
                ),
            )
        )


class SquaredExponentialKernel(Kernel):
    r"""Squared exponential kernel.

    The squared exponential kernel is defined as :math:`k(r_i) = \sigma^2
    \exp\left(-\frac{r_i^2}{2 l_i}\right)` with
    :math:`r = |\mathtt{x1} - \mathtt{x2}|`, kernel variance :math:`\sigma^2`
    and length scales :math:`l`.

    Parameters
    ----------
    lengthscales : (`D`,) array-like
        The length scale :math:`l_i` for each dimension.
    variance : float
        The kernel variance :math:`\sigma^2`.
    """

    def __init__(self, lengthscales, variance=1.0):
        self.lengthscales = np.asarray(lengthscales)
        self.variance = variance

    @property
    def params(self):
        """1d-array of kernel parameters.

        The first `D` values are the length scales for each dimension and the
        last value is the kernel variance.
        """
        return np.concatenate((self.lengthscales, (self.variance,)))

    @params.setter
    def params(self, values):
        self.lengthscales = np.asarray(values[:-1])
        self.variance = values[-1]

    def full(self, x1, x2, what=("y",)):
        res = {}
        d = self._calc_distance(x1, x2)
        res["y"] = self.variance * np.exp(-0.5 * d / self.lengthscales**2)
        if "derivative" in what:
            direction = x1[:, None, :] - x2[None, :, :]
            res["derivative"] = (
                -1.0 / (self.lengthscales**2) * direction * res["y"][:, :, None]
            )
        if "param_derivatives" in what:
            variance_deriv = np.exp(-0.5 * d / self.lengthscales**2)
            lengthscale_deriv = (
                2 * self.variance * d / (self.lengthscales**3) * variance_deriv
            )
            res["param_derivatives"] = [lengthscale_deriv, variance_deriv]
        return res

    def diag(self, x1, x2):
        if x1 is x2:
            return self.variance * np.ones(len(x1))

        d = -2 * np.einsum("ij,ij->i", x1, x2) + (
            np.sum(np.square(x1), 1) + np.sum(np.square(x2), 1)
        )
        return self.variance * np.exp(-0.5 * d / self.lengthscales**2)

    @staticmethod
    def _calc_distance(x1, x2):
        return -2 * np.dot(x1, x2.T) + (
            np.sum(np.square(x1), 1)[:, None] + np.sum(np.square(x2), 1)[None, :]
        )

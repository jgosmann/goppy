"""The GopPy core classes."""

import numpy as np
from numpy.linalg import cholesky, inv

from goppy.growable import GrowableArray


__all__ = ['OnlineGP']


class OnlineGP(object):
    """Online Gaussian Process.

    Provides a Gaussian process to which further data can be efficiently added
    after the initial training.

    Parameters
    ----------
    kernel : kernel object
        Covariance function of the Gaussian process.
    noise_var : float, optional
        The assumed variance of the noise on the training targets.
    expected_size : int, optional
        The overall expected number of training samples to be added to the
        Gaussian process. Setting this parameters can be more efficient memory
        reallocations are avoided.
    buffer_factory : function, optional
        Function to call to create buffer arrays for data storage.

    Attributes
    ----------
    kernel : kernel object
        Covariance function of the Gaussian process.
    noise_var : float, optional
        The assumed variance of the noise on the training targets.
    x_train : (`N`, `D`) ndarray
        The `N` training data inputs of dimension `D`. This will be ``None`` as
        long as the Gaussian process has not been trained.
    y_train : ndarray
        The `N` training data targets of dimension `D`. This will be ``None``
        as long as the Gaussian process has not been trained.
    inv_chol : (`N`, `N`) ndarray
        Inverted lower Cholesky factor of the covariance matrix
        (upper triangular matrix). This will be ``None`` as long as the
        Gaussian process has not been trained.
    trained : bool
        Indicates that the Gaussian process has been fitted to some training
        data.

    Examples
    --------

    >>> from goppy import OnlineGP, SquaredExponentialKernel
    >>> gp = OnlineGP(SquaredExponentialKernel([1.0]), noise_var=0.1)
    >>> gp.fit(np.array([[2, 4]]).T, np.array([[3, 1]]).T)
    >>> gp.add(np.array([[0]]), np.array([[3]]))
    >>> gp.predict(np.array([[1, 3]]).T)
    {'mean': array([[ 2.91154709],
           [ 1.82863199]])}
    """

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

    @property
    def inv_cov_matrix(self):
        """ Inverted covariance matrix.

        Cannot be accessed before the Gaussian process has been trained.
        """
        if self.__inv_cov_matrix is None:
            self.__inv_cov_matrix = np.dot(self.inv_chol.T, self.inv_chol)
        return self.__inv_cov_matrix

    @inv_cov_matrix.deleter
    def inv_cov_matrix(self):
        self.__inv_cov_matrix = None

    def fit(self, x, y):
        """Fits the Gaussian process to training data.

        Parameters
        ----------
        x : (`N`, `D`) array-like
            The `N` input data points of dimension `D` to train on.
        y : (`N`, `D`) array-like
            The `N` training targets with `D` independent dimensions.
        """

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
        """Adds additional training data to the Gaussian process and adjusts
        the fit.

        Parameters
        ----------
        x : (`N`, `D`) array-like
            The `N` input data points of dimension `D` to train on.
        y : (`N`, `D`) array-like
            The `N` training targets with `D` independent dimensions.
        """

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
        r"""Predict with the Gaussian process.

        Depending on the values included in the `what` parameter different
        predictions will be made:

        * ``'mean'``: Mean prediction of the Gaussian process of shape (`N`,
          `D`).
        * ``'mse'``: Predictive variance of the Gaussian process of shape
          (`N`,).
        * ``'derivative'``: Predicted derivative of the mean.
          ``res['derivative'][i, :, j]`` will correspond to
          :math:`\left(\frac{\partial \mu}{\partial x_j}\right)
          \left(x_i\right)` with the `i`-th input data point :math:`x_i`,
          and mean function :math:`\mu(x)`.
        * ``'mse_derivative'``: Predicted derivative of the variance.
          ``res['mse_derivative'][i, :]`` will correspond to
          :math:`\left(\frac{d \sigma^2}{d x}\right)
          \left(x_i\right)` with the `i`-th input data point :math:`x_i`,
          and variance function :math:`\sigma^2(x)`.

        Parameters
        ----------
        x : (`N`, `D`) array-like
            The `N` data points of dimension `D` to predict data for.
        what : set-like, optional
            Types of predictions to be made (see above).

        Returns
        -------
        dict
            Dictionary with the elements of `what` as keys and the
            corresponding predictions as values.
        """
        pred = {}

        if 'derivative' in what or 'mse_derivative' in what:
            kernel_what = ('y', 'derivative')
        else:
            kernel_what = ('y',)

        lazy_vars = _LazyVarCollection(
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

    def calc_log_likelihood(self, what=('value',)):
        r"""Calculate the log likelihood or its derivative of the Gaussian
        process.

        Depending on the values included in the `what` parameter different
        values will be calculated:

        * ``'value'``: The log likelihood of the Gaussian process as scalar.
        * ``'derivative'``: Partial derivatives of the log likelihood for each
            kernel parameter as array. See the ``params`` property of the used
            kernel for the order.

        Parameters
        ----------
        what : set-like, optional
            Values to calculate (see above).

        Returns
        -------
        dict
            Dictionary with the elements of `what` as keys and the
            corresponding calculated values.
        """
        res = {}
        svs = np.dot(self.inv_chol, self.y_train)
        if 'value' in what:
            res['value'] = np.squeeze(
                -0.5 * np.dot(svs.T, svs) +
                np.sum(np.log(np.diag(self.inv_chol))) -
                0.5 * len(self.y_train) * np.log(2 * np.pi))
        if 'derivative' in what:
            alpha = np.dot(self.inv_chol.T, svs)
            grad_weighting = np.dot(alpha, alpha.T) - self.inv_cov_matrix
            res['derivative'] = np.array([
                0.5 * np.sum(np.einsum(
                    'ij,ji->i', grad_weighting, param_deriv))
                for param_deriv in self.kernel.full(
                    self.x_train, self.x_train, what='param_derivatives')[
                    'param_derivatives']])
        return res


class _LazyVarCollection(object):
    def __init__(self, **kwargs):
        self._eval_fns = kwargs

    def __getattr__(self, name):
        value = self._eval_fns[name](self)
        setattr(self, name, value)
        return value

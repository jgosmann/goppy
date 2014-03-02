"""Unit tests for kernel module."""

from hamcrest import assert_that, is_, equal_to
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal

from ..kernel import SquaredExponentialKernel


class TestSquaredExponentialKernel(object):

    datasets = [
        {
            'x1': np.array([[1, 1, 1], [1, 2, 1]]),
            'x2': np.array([[1, 2, 3], [4, 2, 1]]),
            'params': {
                'lengthscales': [0.6],
                'variance': 0.75
            },
            'y': np.array([
                [0.00072298179430063214, 6.9693689985354893e-07],
                [0.00289944010460460330, 2.7949898790590032e-06]]),
            'derivative': np.array([
                [[0.0, 0.00200828, 0.00401657],
                 [5.80780750e-06, 1.93593583e-06, 0.0]],
                [[0.0, 0.0, 0.016108],
                 [2.32915823e-05, 0.00000000e+00, 0.00000000e+00]]]),
            'param_derivatives': [
                np.array([[3.34713794e-02, 6.45311944e-05],
                          [1.07386671e-01, 2.32915823e-04]]),
                np.array([[9.63975726e-04, 9.29249200e-07],
                          [3.86592014e-03, 3.72665317e-06]])
            ]
        }
    ]

    def test_kernel(self):
        for dataset in self.datasets:
            yield self.check_kernel, dataset

    def check_kernel(self, dataset):
        kernel = SquaredExponentialKernel(**dataset['params'])
        x1, x2 = (dataset['x1'], dataset['x2'])
        result = kernel.full(x1, x2, what=(
            'y', 'derivative', 'param_derivatives'))
        print result['param_derivatives']
        assert_almost_equal(result['y'], dataset['y'])
        assert_almost_equal(result['derivative'], dataset['derivative'])
        assert_almost_equal(
            result['param_derivatives'], dataset['param_derivatives'])

    def test_kernel_can_be_used_as_function(self):
        for dataset in self.datasets:
            yield self.check_can_be_used_as_function, dataset

    def check_can_be_used_as_function(self, dataset):
        kernel = SquaredExponentialKernel(**dataset['params'])
        x1, x2 = (dataset['x1'], dataset['x2'])
        assert_equal(kernel(x1, x2), kernel.full(x1, x2)['y'])

    def test_diag(self):
        for dataset in self.datasets:
            yield self.check_diag, dataset

    def check_diag(self, dataset):
        kernel = SquaredExponentialKernel(**dataset['params'])
        x1, x2 = (dataset['x1'], dataset['x2'])
        assert_equal(kernel.diag(x1, x2), np.diag(kernel(x1, x2)))

    # TODO refactor: too much code duplication for iterating over data sets.
    def test_diag_symmetric(self):
        for dataset in self.datasets:
            yield self.check_diag_symmetric, dataset

    def check_diag_symmetric(self, dataset):
        kernel = SquaredExponentialKernel(**dataset['params'])
        x = dataset['x1']
        expected = dataset['params']['variance'] * np.ones(len(x))
        assert_equal(kernel.diag(x, x), expected)

    def test_can_get_params_as_array(self):
        kernel = SquaredExponentialKernel(**self.datasets[0]['params'])
        assert_equal(kernel.params, np.concatenate((
            self.datasets[0]['params']['lengthscales'],
            (self.datasets[0]['params']['variance'],))))

    def test_can_set_params_as_array(self):
        kernel = SquaredExponentialKernel(**self.datasets[0]['params'])
        kernel.params = np.array([1.2, 0.5])
        assert_equal(kernel.lengthscales, [1.2])
        assert_that(kernel.variance, is_(equal_to(0.5)))

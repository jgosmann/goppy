"""Unit tests for goppy module."""

from hamcrest import assert_that, close_to, contains_inanyorder, is_
import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
from mock import ANY, call, MagicMock

from ..core import _LazyVarCollection, OnlineGP
from ..kernel import SquaredExponentialKernel


class GPBuilder(object):
    def __init__(self):
        self.kernel = SquaredExponentialKernel([1.0])
        self.noise_var = 0.01
        self.expected_size = None
        self.buffer_factory = None

    def with_kernel(self, kernel):
        self.kernel = kernel
        return self

    def with_noise_var(self, noise_var):
        self.noise_var = noise_var
        return self

    def with_expected_size(self, size):
        self.expected_size = size
        return self

    def with_training_config(self, training_config):
        return (self
                .with_kernel(training_config['kernel'])
                .with_noise_var(training_config['noise_var']))

    def with_buffer_factory(self, factory):
        self.buffer_factory = factory
        return self

    def build(self):
        kwargs = {}
        if self.expected_size is not None:
            kwargs['expected_size'] = self.expected_size
        if self.buffer_factory is not None:
            kwargs['buffer_factory'] = self.buffer_factory
        return OnlineGP(self.kernel, self.noise_var, **kwargs)


class TestOnlineGP(object):

    datasets = [
        {
            # normal dataset
            'training': {
                'X': np.array([[-4, -2, -0.5, 0, 2]]).T,
                'Y': np.array([[-2, 0, 1, 2, -1]]).T,
                'kernel': SquaredExponentialKernel([1.0]),
                'noise_var': 0.5
            },
            'tests': [
                {
                    'X': np.array([[-3, 1]]).T,
                    'Y': np.array([[-0.78511166, 0.37396387]]).T,
                    'mse': np.array([1.04585738, 1.04888027]),
                    'derivative': np.array([[[0.85538797]], [[-1.30833924]]]),
                    'mse_derivative': np.array([[-0.00352932], [-0.00173095]])
                }
            ],
            'log_likelihood': -8.51911832,
            'log_likelihood_derivative': np.array([-0.76088728, 0.49230927])

        }, {
            # data as lists
            'training': {
                'X': [[-4], [-2], [-0.5], [0], [2]],
                'Y': [[-2], [0], [1], [2], [-1]],
                'kernel': SquaredExponentialKernel([1.0]),
                'noise_var': 0.5
            },
            'tests': [
                {
                    'X': np.array([[-3, 1]]).T,
                    'Y': np.array([[-0.78511166, 0.37396387]]).T,
                    'mse': np.array([1.04585738, 1.04888027]),
                    'derivative': np.array([[[0.85538797]], [[-1.30833924]]]),
                    'mse_derivative': np.array([[-0.00352932], [-0.00173095]])
                }
            ],
            'log_likelihood': -8.51911832,
            'log_likelihood_derivative': np.array([-0.76088728, 0.49230927])
        }
    ]

    def test_prediction(self):
        for dataset in self.datasets:
            for test in dataset['tests']:
                yield self.check_prediction, dataset['training'], test

    def check_prediction(self, training, test):
        gp = GPBuilder().with_training_config(training).build()
        gp.fit(training['X'], training['Y'])
        self._assert_prediction_matches_data(gp, test)

    def test_adding_data_online(self):
        for dataset in self.datasets:
            training = dataset['training']
            gp = GPBuilder().with_training_config(training).build()
            for x, y in zip(training['X'], training['Y']):
                gp.add([x], [y])

            for test in dataset['tests']:
                yield self._assert_prediction_matches_data, gp, test

    @staticmethod
    def _assert_prediction_matches_data(gp, data):
        pred = gp.predict(
            data['X'], what=['mean', 'mse', 'derivative', 'mse_derivative'])
        assert_almost_equal(pred['mean'], data['Y'])
        assert_almost_equal(pred['mse'], data['mse'])
        assert_almost_equal(pred['derivative'], data['derivative'])
        assert_almost_equal(pred['mse_derivative'], data['mse_derivative'])

    def test_allows_adding_empty_datasets(self):
        gp = GPBuilder().build()
        data = self.datasets[0]['training']
        gp.fit(data['X'], data['Y'])
        expected = gp.inv_cov_matrix
        gp.add([], [])
        actual = gp.inv_cov_matrix
        assert_equal(actual, expected)

    def test_has_trained_indicator(self):
        gp = GPBuilder().build()
        assert_that(gp.trained, is_(False))
        data = self.datasets[0]['training']
        gp.fit(data['X'], data['Y'])
        assert_that(gp.trained, is_(True))

    def test_uses_expected_size(self):
        size = 30
        factory = MagicMock()
        factory.side_effect = lambda *args, **kwargs: MagicMock()
        gp = (GPBuilder()
              .with_buffer_factory(factory)
              .with_expected_size(size)
              .build())
        data = self.datasets[0]['training']
        gp.fit(data['X'], data['Y'])
        expected_calls = [
            call(ANY, buffer_shape=(size,)),
            call(ANY, buffer_shape=(size,)),
            call(ANY, buffer_shape=(size, size))]
        assert_that(factory.mock_calls, contains_inanyorder(*expected_calls))

    def test_likelihood(self):
        for dataset in self.datasets:
            yield self.check_likelihood, dataset['training'], \
                dataset['log_likelihood']

    def check_likelihood(self, training, log_likelihood):
        gp = GPBuilder().with_training_config(training).build()
        gp.fit(training['X'], training['Y'])
        assert_that(
            gp.calc_log_likelihood()['value'],
            is_(close_to(log_likelihood, 1e-6)))

    def test_likelihood_derivative(self):
        for dataset in self.datasets:
            yield self.check_likelihood_derivative, dataset['training'], \
                dataset['log_likelihood_derivative']

    def check_likelihood_derivative(self, training, derivative):
        gp = GPBuilder().with_training_config(training).build()
        gp.fit(training['X'], training['Y'])
        assert_almost_equal(gp.calc_log_likelihood(
            what=('derivative',))['derivative'], derivative)


class TestLazyVarCollection(object):
    def test_returns_function_return_value_on_var_request(self):
        var_collection = _LazyVarCollection(test_var=lambda self: 23)
        assert_that(var_collection.test_var, is_(23))

    def test_function_call_is_lazy(self):
        mock = MagicMock()
        var_collection = _LazyVarCollection(test_var=mock)
        assert_that(mock.called, is_(False))
        # pylint: disable=pointless-statement
        var_collection.test_var
        assert_that(mock.called, is_(True))

    def test_caches_function_result(self):
        mock = MagicMock()
        mock.return_value = 23
        var_collection = _LazyVarCollection(test_var=mock)
        assert_that(var_collection.test_var, is_(mock.return_value))
        assert_that(var_collection.test_var, is_(mock.return_value))
        mock.assert_called_once_with(var_collection)

    def test_allows_chaining(self):
        var_collection = _LazyVarCollection(
            var1=lambda self: 2, var2=lambda self: 3 * self.var1)
        assert_that(var_collection.var2, is_(6))

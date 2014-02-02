"""Unit tests for goppy module."""

import numpy as np
from numpy.testing import assert_almost_equal

from ..goppy import OnlineGP
from ..goppy import SquaredExponentialKernel


class TestOnlineGP(object):

    datasets = [
        {
            'training': {
                'X': np.array([[-4, -2, -0.5, 0, 2]]).T,
                'Y': np.array([[-2, 0, 1, 2, -1]]).T
            },
            'tests': [
                {
                    'kernel': SquaredExponentialKernel([1.0]),
                    'noise_var': 0.5,
                    'X': np.array([[-3, 1]]).T,
                    'Y': np.array([[-0.78511166, 0.37396387]]).T
                }
            ]
        }
    ]

    def test_prediction(self):
        for dataset in self.datasets:
            for test in dataset['tests']:
                yield self.check_prediction, dataset['training'], test

    def check_prediction(self, training, test):
        gp = OnlineGP(test['kernel'], test['noise_var'])
        gp.fit(training['X'], training['Y'])
        prediction = gp.predict(test['X'])
        assert_almost_equal(prediction, test['Y'])

    #def setUp(self):
        #self.gp = plume.prediction.OnlineGP(
            #plume.prediction.RBFKernel(1.0), noise_var=0.5)

    #def test_can_predict(self):
        #x = np.array([[-4, -2, -0.5, 0, 2]]).T
        #y = np.array([[-2, 0, 1, 2, -1]]).T
        #self.gp.fit(x, y)

        #x_star = np.array([[-3, 1]]).T
        #expected = np.array([[-0.78511166, 0.37396387]]).T
        #pred = self.gp.predict(x_star)
        #assert_almost_equal(pred, expected)

    #def test_evaluates_mse(self):
        #x = np.array([[-4, -2, -0.5, 0, 2]]).T
        #y = np.array([[-2, 0, 1, 2, -1]]).T
        #self.gp.fit(x, y)

        #x_star = np.array([[-3, 1]]).T
        #expected = [1.04585738, 1.04888027]
        #unused, mse = self.gp.predict(x_star, eval_MSE=True)
        #assert_almost_equal(mse, expected)

    #def test_allows_adding_new_datapoints_online(self):
        #xs = [-4, -2, -0.5, 0, 2]
        #ys = [-2, 0, 1, 2, -1]

        #for x, y, in zip(xs, ys):
            #self.gp.add_observations(np.array([[x]]), np.array([[y]]))

        #x_star = np.array([[-3, 1]]).T
        #expected = np.array([[-0.78511166, 0.37396387]]).T
        #expected_mse = [1.04585738, 1.04888027]
        #pred, mse = self.gp.predict(x_star, eval_MSE=True)
        #assert_almost_equal(pred, expected)
        #assert_almost_equal(mse, expected_mse)

    #def test_has_trained_indicator(self):
        #assert_that(self.gp.trained, is_(False))
        #x = np.array([[-4, -2, -0.5, 0, 2]]).T
        #y = np.array([[-2, 0, 1, 2, -1]]).T
        #self.gp.fit(x, y)
        #assert_that(self.gp.trained, is_(True))

    #def test_can_calculate_prediction_derivative(self):
        #x = np.array([[-4, -2, -0.5, 0, 2]]).T
        #y = np.array([[-2, 0, 1, 2, -1]]).T
        #self.gp.fit(x, y)

        #x_star = np.array([[-3, 1]]).T
        #expected = np.array([[[0.85538797]], [[-1.30833924]]])
        #unused, actual = self.gp.predict(x_star, eval_derivatives=True)
        #assert_almost_equal(actual, expected)

    #def test_can_calculate_mse_derivative(self):
        #x = np.array([[-4, -2, -0.5, 0, 2]]).T
        #y = np.array([[-2, 0, 1, 2, -1]]).T
        #self.gp.fit(x, y)

        #x_star = np.array([[-3, 1]]).T
        #expected = np.array([[-0.00352932], [-0.00173095]])
        #unused, unused, unused, actual = self.gp.predict(
            #x_star, eval_MSE=True, eval_derivatives=True)
        #assert_almost_equal(actual, expected)

    #def test_can_calculate_neg_log_likelihood(self):
        #x = np.array([[-4, -2, -0.5, 0, 2]]).T
        #y = np.array([[-2, 0, 1, 2, -1]]).T
        #self.gp.fit(x, y)
        #actual = self.gp.calc_neg_log_likelihood()
        #expected = 8.51911832
        #assert_almost_equal(actual, expected)

    #def test_can_calculate_neg_log_likelihood_derivative(self):
        #x = np.array([[-4, -2, -0.5, 0, 2]]).T
        #y = np.array([[-2, 0, 1, 2, -1]]).T
        #self.gp.fit(x, y)
        #actual = self.gp.calc_neg_log_likelihood(eval_derivative=True)
        #expected = np.array([0.76088728, -0.49230927])
        #assert_almost_equal(actual[1], expected)

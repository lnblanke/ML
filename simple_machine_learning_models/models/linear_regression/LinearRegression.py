# Aggregate linear regression models
# @Time: 3/1/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: LinearRegression.py
import numpy as np

from .BatchGradientDescend import BatchGradientDescend
from .StochasticGradientDescend import StochasticGradientDescend
from .NormalEquation import NormalEquation
from .LocallyWeightedRegression import LocallyWeightedRegression
from ..Regressor import Regressor


class LinearRegression(Regressor):
    name = "linear regression"

    def __init__(self, type: str, n_features, learning_rate = 0.1):
        super().__init__(n_features)
        self.rate = learning_rate

        type = type.lower()

        if type == "bgd" or type == "batch gradient descend":
            self._model = BatchGradientDescend(self.n_features + 1, self.rate)
        elif type == "sgd" or type == "stochastic gradient descend":
            self._model = StochasticGradientDescend(self.n_features + 1, self.rate)
        elif type == "ne" or type == "normal equation":
            self._model = NormalEquation(self.n_features + 1)
        elif type == "lwr" or type == "locally weighted regression":
            self._model = LocallyWeightedRegression(self.n_features + 1, self.rate)
        else:
            raise TypeError(f"The type {type} is not found!")

    def train(self, train_x, train_y, verbose = 1):
        train_x = np.insert(train_x, len(train_x[0]), np.ones(len(train_y)), axis = 1)

        self._model.train(train_x, train_y, verbose)

    def predict(self, test_x):
        test_x = np.insert(test_x, len(test_x[0]), np.ones(len(test_x)), axis = 1)
        return self._model.predict(test_x)

# Aggregate linear regression models
# @Time: 3/1/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: linear_regression.py

from batch_gradient_descend import BatchGradientDescend
from stochastic_gradient_descend import StochasticGradientDescend
from normal_equation import NormalEquation
from locally_weighted_regression import LocallyWeightedRegression

class LinearRegression:
    def __init__(self, type, n_feature, learning_rate = 0.1):
        self.n_feature = n_feature
        self.rate = learning_rate

        if type == "BGD" or type == "Batch Gradient Descend":
            self._model = BatchGradientDescend(self.n_feature, self.rate)
        elif type == "SGD" or type == "Stochastic Gradient Descend":
            self._model = StochasticGradientDescend(self.n_feature, self.rate)
        elif type == "NE" or type == "Normal Equation":
            self._model = NormalEquation(self.n_feature)
        elif type == "LWR" or type == "Locally Weighted Regression":
            self._model = LocallyWeightedRegression(self.n_feature)
        else:
            raise TypeError(f"The type {type} is not found!")

        print(self._model.__class__)

    def train(self, train_x, train_y):
        self._model.train(train_x, train_y)

    def predict(self, test_x):
        return self._model.predict(test_x)

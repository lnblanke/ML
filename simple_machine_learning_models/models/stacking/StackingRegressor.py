# A stacking regressor basing on decision tree regressor and linear regression
# @Time: 11/5/21
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: StackingRegressor.py.py

from ..linear_regression import LinearRegression
from ..decision_tree import DecisionTreeRegressor
import numpy as np
from ..tools import bootstrap
from ..Ensemble import Ensemble
from ..Regressor import Regressor
import warnings


class StackingRegressor(Ensemble, Regressor):
    name = "stacking regressor"

    def __init__(self, n_features, n_predictor: int, max_depth = 3):
        super().__init__(n_features, n_predictor, DecisionTreeRegressor, n_features, max_depth)

        self.blender = LinearRegression(self.n_predictor, "NE")

    def train(self, train_x, train_y):
        subset1_x = np.array(train_x[:len(train_y) // 2])
        subset1_y = np.array(train_y[:len(train_y) // 2])
        subset2_x = np.array(train_x[len(train_y) // 2 + 1:])
        subset2_y = np.array(train_y[len(train_y) // 2 + 1:])

        pred = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for predictor in self.predictors:
                x, y = bootstrap(subset1_x, subset1_y, len(subset1_y))
                predictor.train(x, y)
                pred.append(predictor.predict(subset2_x))

        pred = np.transpose(np.array(pred))

        self.blender.train(pred, subset2_y)

    def predict(self, test_x):
        pred = []

        for predictor in self.predictors:
            pred.append(predictor.predict(test_x))

        pred = np.transpose(np.array(pred))

        return self.blender.predict(pred)

# A simple random forest regressor using decision tree regressor
# @Time: 11/04/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: RandomForestRegressor.py
import numpy as np
from ..decision_tree import DecisionTreeRegressor
import warnings
from ..tools import bootstrap
from ..Ensemble import Ensemble
from ..Regressor import Regressor


class RandomForestRegressor(Ensemble, Regressor):
    name = "random forest regressor"

    def __init__(self, n_features, n_predictor: int, max_depth = 3):
        super().__init__(n_features, n_predictor, DecisionTreeRegressor, n_features, max_depth)

    def train(self, train_x, train_y):
        for tree in self.predictors:
            sample_x, sample_y = bootstrap(train_x, train_y, len(train_y))

            # Deal with potential RuntimeWarning raised by NumPy
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tree.train(sample_x, sample_y)

    def predict(self, test_x):
        sep_pred = []

        for tree in self.predictors:
            sep_pred.append(tree.predict(test_x))

        pred = np.mean(sep_pred, axis = 0)

        return pred

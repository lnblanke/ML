# A simple gradient boosting regression tree model
# @Time: 10/28/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: GradBoost.py

import numpy as np
from ..decision_tree import DecisionTreeRegressor
from ..Ensemble import Ensemble
from ..Regressor import Regressor


class GradBoost(Ensemble, Regressor):
    name = "gradient boost"

    def __init__(self, n_features, n_predictor = 10, max_depth = 3):
        super().__init__(n_features, n_predictor, DecisionTreeRegressor, n_features, max_depth)

    def train(self, train_x, train_y):
        y = np.copy(train_y)
        for i in range(self.n_predictor):
            self.predictors[i].train(train_x, y)

            pred = self.predictors[i].predict(train_x)

            print("Epoch: {:d} MSE: {:f}".format(i + 1, np.sum((y - pred) ** 2) / len(y)))

            y -= pred

    def predict(self, test_x):
        pred = np.sum(predictor.predict(test_x) for predictor in self.predictors)

        return pred

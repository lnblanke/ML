# A simple random forest classifier using decision tree classifier
# @Time: 9/23/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: RandomForestClassifier.py
import numpy as np
from ..decision_tree import DecisionTreeClassifier
from tools import bootstrap
from ..Ensemble import Ensemble
from ..Classifier import Classifier


class RandomForestClassifier(Ensemble, Classifier):
    name = "random forest classifier"

    def __init__(self, n_features, n_predictor = 10, max_depth = 1):
        super().__init__(n_features, n_predictor, DecisionTreeClassifier, n_features, max_depth)

    def train(self, train_x, train_y):
        for tree in self.predictors:
            sample_x, sample_y = bootstrap(train_x, train_y, len(train_y))

            tree.train(sample_x, sample_y)

    def predict(self, test_x):
        sep_pred = []

        for tree in self.predictors:
            sep_pred.append(tree.predict(test_x))

        pred = np.max(sep_pred, axis = 0)

        return pred

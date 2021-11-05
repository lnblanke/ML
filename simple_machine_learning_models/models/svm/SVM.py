# A simple support vector machine for classification using scikit-learn model
# @Time: 3/21/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: SVM.py

import numpy as np
from ..Classifier import Classifier


class SVM(Classifier):
    def __init__(self, n_features):
        super().__init__(n_features)

    def train(self, train_x: np.ndarray, train_y: np.ndarray):
        raise NotImplementedError()

    def predict(self, test_x: np.ndarray):
        raise NotImplementedError()

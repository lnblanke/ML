# Supervised model prototype
# @Time: 11/5/21
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Supervised.py.py

import abc

import numpy as np

from .Model import Model


class Supervised(Model):
    name = "supervised model"

    def __init__(self, n_features):
        super().__init__(n_features)

    @abc.abstractmethod
    def train(self, train_x: np.ndarray, train_y: np.ndarray):
        pass

    @abc.abstractmethod
    def predict(self, test_x: np.ndarray):
        pass

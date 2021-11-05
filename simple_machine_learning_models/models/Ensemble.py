# Ensemble model prototype
# @Time: 11/5/21
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Ensemble.py.py

from .Model import Model


class Ensemble(Model):
    name = "ensemble model"

    def __init__(self, n_features, n_predictor: int, predictor = None, *args):
        super().__init__(n_features)
        self.n_predictor = n_predictor
        self.predictors = []

        for i in range(self.n_predictor):
            self.predictors.append(predictor(*args))

# A simple EM unsupervised classification model
# @Time: 6/20/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: EM.py

from ..Unsupervised import Unsupervised


class EM(Unsupervised):
    def __init__(self, n_features):
        super().__init__(n_features)

    def train(self, train_x):
        raise NotImplementedError()

# Classifier prototype
# @Time: 11/5/21
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Classifier.py.py

import abc
from .Supervised import Supervised


class Classifier(Supervised):
    name = "classifier"

    def __init__(self, n_features):
        super().__init__(n_features)

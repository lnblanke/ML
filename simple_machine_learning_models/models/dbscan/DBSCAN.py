# A simple DBSCAN model for clustering
# @Time: 11/12/21
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: DBSCAN.py.py

import numpy as np
from ..Unsupervised import Unsupervised


class DBSCAN(Unsupervised):
    name = "DBSCAN"

    def __init__(self, n_features, eps: float, min_samples: int):
        super().__init__(n_features)
        self.eps = eps
        self.min_samples = min_samples

    def _find_core(self, color, i):
        if color[i] == i:
            return i

        color[i] = self._find_core(color, color[i])

        return color[i]

    def train(self, train_x: np.ndarray):
        color = np.arange(0, len(train_x))

        for i in range(len(train_x)):
            neighbors = [j for j in range(len(train_x)) if
                         i != j and np.linalg.norm(train_x[i] - train_x[j]) <= self.eps]

            if len(neighbors) < self.min_samples:
                continue

            for neighbor in neighbors:
                core = self._find_core(color, neighbor)
                color[core] = self._find_core(color, i)

        indice = {}

        for i in range(len(color)):
            if color[i] not in indice:
                indice[color[i]] = 0

            indice[color[i]] += 1

        idx = 0

        for index in indice:
            if indice[index] == 1:
                indice[index] = -1
            else:
                indice[index] = idx
                idx += 1

        for i in range(len(color)):
            color[i] = indice[color[i]]

        return color

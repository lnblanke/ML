# A simple K-means classifier for unsupervised learning clustering
# @Time: 3/28/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: KMeans.py

import numpy as np
from ..Unsupervised import Unsupervised


class KMeans(Unsupervised):
    name = "K-Means"

    def __init__(self, n_features, cluster: int):
        super().__init__(n_features)
        self.n_clusters = cluster

    def train(self, train_x):
        p = np.zeros((self.n_clusters, self.n_features))

        for i in range(self.n_clusters):
            p[i] = train_x[i]

        color = np.zeros(len(train_x))

        count = 0

        _cost = 0

        while True:
            for i in range(len(train_x)):
                color[i] = np.argmin([np.linalg.norm(train_x[i] - p[0]), np.linalg.norm(train_x[i] - p[1])])

            for i in range(self.n_clusters):
                up = 0
                down = 0
                for j in range(len(train_x)):
                    up += (color[j] == i) * train_x[j]
                    down += (color[j] == i)

                p[i] = up / down

            cost = 0
            count += 1
            for i in range(len(train_x)):
                cost += np.linalg.norm(train_x[i] - p[int(color[i])])

            print("Epoch: %d Loss: %.5f" % (count, cost))

            if cost == _cost:
                break

            _cost = cost

        return color

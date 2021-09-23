# A simple K-means classifier for unsupervised learning clustering
# @Time: 3/28/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Kmeans.py

import numpy as np

class kmeans:
    def __init__(self, cluster):
        self.num_clusters = cluster

    def train(self, train_data):
        p = np.zeros((self.num_clusters, len(train_data[0])))

        for i in range(self.num_clusters):
            p[i] = train_data[i]

        color = np.zeros(len(train_data))

        count = 0

        _cost = 0

        while True:
            for i in range(len(train_data)):
                color[i] = np.argmin([np.linalg.norm(train_data[i] - p[0]), np.linalg.norm(train_data[i] - p[1])])

            for i in range(self.num_clusters):
                up = 0
                down = 0
                for j in range(len(train_data)):
                    up += (color[j] == i) * train_data[j]
                    down += (color[j] == i)

                p[i] = up / down

            cost = 0
            count += 1
            for i in range(len(train_data)):
                cost += np.linalg.norm(train_data[i] - p[int(color[i])])

            print("Epoch: %d Loss: %.5f" % (count, cost))

            if cost == _cost:
                break

            _cost = cost

        return color

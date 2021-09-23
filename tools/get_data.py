# Get classification data from sklearn
# @Time: 9/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: get_classification_data.py

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

def get_classification_data(samples = 1000, classes = 2, clusters = 1, features = 2, sep = 1, split = .2,
                            supervised = True):
    data = make_classification(n_samples = samples, n_classes = classes, n_clusters_per_class = clusters,
        n_features = features, n_redundant = 0, class_sep = sep, n_informative = 2)

    if not supervised:
        return data[0], data[1]

    train_x, test_x, train_y, test_y = train_test_split(data[0], data[1], test_size = split)

    return train_x, test_x, train_y, test_y

def get_regression_data(samples = 1000, features = 1, bias = 2, noise = 15, split = .2):
    data = make_regression(n_samples = samples, n_features = features, bias = bias, noise = noise)
    train_x, test_x, train_y, test_y = train_test_split(data[0], data[1], test_size = split)

    return train_x, test_x, train_y, test_y
# Train simple machine learning models
# @Time: 10/7/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: train.py.py

from models import *
from tools import *
import time
import numpy as np

n_samples = 1000  # # of samples
n_features = 2  # # of features
max_depth = 5  # Max depth for decision tree
n_classes = 4  # # of classes for softmax regression
n_predictor = 10  # # of classifiers for ensemble models
n_clusters = 2  # # of clusters for unsupervised models

if __name__ == '__main__':
    while 1:
        name = input("Please input the model to train: ")
        model_selected = None

        for m in models:
            if name.lower().replace(" ", "") == m.name.lower().replace(" ", ""):
                model_selected = m
                break

        if model_selected is None:
            print("The model you entered does not exist!")
        else:
            break

    init = time.time()

    if model_selected is DecisionTreeClassifier or model_selected is DecisionTreeRegressor:
        model = model_selected(n_features, max_depth)
    elif model_selected is DBSCAN:
        model = model_selected(n_features, .3, 5)
    elif issubclass(model_selected, Ensemble):
        model = model_selected(n_features, n_predictor)
    elif model_selected is SoftmaxRegression:
        model = model_selected(n_features, n_classes)
    elif model_selected is KMeans:
        model = model_selected(n_features, n_clusters)
    elif model_selected is not LinearRegression:
        model = model_selected(n_features)
    else:
        while True:
            try:
                model = model_selected(input("Please input the type of linear regression model: "), n_features)
                break
            except TypeError:
                print("The model does not exist!")

    print(f"Training {model_selected.name} model...")

    if isinstance(model, Classifier):
        if model_selected is not SoftmaxRegression:
            train_x, test_x, train_y, test_y = get_classification_data(samples = n_samples, features = n_features,
                                                                       sep = 1.5,
                                                                       clusters = 2)
        else:
            train_x, test_x, train_y, test_y = get_classification_data(samples = n_samples, features = n_features,
                                                                       classes = n_classes, sep = 1.5)

        model.train(train_x, train_y)
        pred = model.predict(test_x)

        print("Accuracy: %.2f" % (np.sum(pred == test_y) / len(test_y) * 100) + "%")
        print("Time: %.5fms" % ((time.time() - init) * 1000))

        show(test_x, pred, test_y, model.name)
    elif isinstance(model, Regressor):
        train_x, test_x, train_y, test_y = get_regression_data(samples = n_samples, features = n_features)

        model.train(train_x, train_y)
        pred = model.predict(test_x)

        mse = np.sum((pred - test_y) ** 2)

        print("Loss:", mse / len(test_y))
        print("Time: %.5fms" % ((time.time() - init) * 1000))

        if n_features == 1:
            show_trendline(test_x, test_y, model, model.name)
    elif isinstance(model, Unsupervised):
        x, y = get_classification_data(samples = n_samples, features = n_features, sep = 1.5, supervised = False)

        pred = model.train(x)
        print("Time: %.5fms" % ((time.time() - init) * 1000))

        show(x, pred, None, model.name)

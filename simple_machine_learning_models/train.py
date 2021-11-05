# Train simple machine learning models
# @Time: 10/7/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: train.py.py

from models import *
from tools import *
import time
import numpy as np

n_samples = 1000 # # of samples
n_features = 2 # # of features
max_depth = 10 # Max depth for decision tree
n_classes = 4 # # of classes for softmax regression
n_trees = 10 # # of trees for random forest

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

    print(f"Training {model_selected.name} model...")

    init = time.time()

    if model_selected is DecisionTreeClassification:
        model = model_selected(n_features, max_depth)
    elif model_selected is DecisionTreeRegression:
        model = model_selected(1, max_depth // 3)
    elif model_selected is GradBoost:
        model = model_selected(1)
    elif model_selected is RandomForest:
        model = model_selected(n_features, n_trees)
    elif model_selected is SoftmaxRegression:
        model = model_selected(n_features, n_classes)
    elif model_selected is not LinearRegression:
        model = model_selected(n_features)
    else:
        model = None

    supervised = supervised_models.count(model_selected) > 0

    classification = classification_models.count(model_selected) > 0

    if supervised and classification:
        if model_selected is not SoftmaxRegression:
            train_x, test_x, train_y, test_y = get_classification_data(samples = n_samples, features = n_features, sep = 1.5, clusters = 2)
        else:
            train_x, test_x, train_y, test_y = get_classification_data(samples = n_samples, features = n_features, classes = n_classes, sep = 1.5)

        model.train(train_x, train_y)
        pred = model.predict(test_x)

        print("Accuracy: %.2f" % (np.sum(pred == test_y) / len(test_y) * 100) + "%")
        print("Time: %.5fms" % ((time.time() - init) * 1000))

        show(test_x, pred, test_y, model.name)
    elif supervised and not classification:
        train_x, test_x, train_y, test_y = get_regression_data(samples = n_samples)

        if model_selected is LinearRegression:
            type_list = ["Batch Gradient Descend", "Stochastic Gradient Descend", "Normal Equation", "Locally Weighted Regression"]

            for type in type_list:
                init = time.time()

                model = LinearRegression(type, len(train_x[0]))

                print(f"{type}:")

                model.train(train_x, train_y)

                pred = model.predict(test_x)

                mse = np.sum((pred - test_y) ** 2)

                print("Loss:", mse / len(test_y))
                print("Time: %.5fms" % ((time.time() - init) * 1000))

                show_trendline(test_x, test_y, model, type)
        else:
            model.train(train_x, train_y)
            pred = model.predict(test_x)

            mse = np.sum((pred - test_y) ** 2)

            print("Loss:", mse / len(test_y))
            print("Time: %.5fms" % ((time.time() - init) * 1000))

            show_trendline(test_x, test_y, model, model.name)
    elif not supervised:
        x, y = get_classification_data(samples = n_samples, features = n_features, sep = 1.5, supervised = False)

        pred = model.train(x)
        print("Time: %.5fms" % ((time.time() - init) * 1000))

        show(x, pred, y, model.name)

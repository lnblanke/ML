# Train simple machine learning models
# @Time: 10/7/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: train.py.py

from models import *
import time
import numpy as np


def train(model_name: str, *args, n_samples = 1000, n_features = 2, sep = 1.5, n_clusters = 2, random_state = None):
    model_selected = None

    for m in models:
        if model_name.lower().replace(" ", "") == m.name.lower().replace(" ", ""):
            model_selected = m
            break

    if model_selected is None:
        raise NameError("The model you entered does not exist!")

    init = time.time()

    model = model_selected(n_features, *args)

    print(f"Training {model_selected.name} model...")

    if isinstance(model, Classifier):
        # Classification models
        if model_selected is not SoftmaxRegression:
            train_x, test_x, train_y, test_y = get_classification_data(samples = n_samples, features = n_features,
                                                                       sep = sep,
                                                                       clusters = n_clusters,
                                                                       random_state = random_state)
        else:
            train_x, test_x, train_y, test_y = get_classification_data(samples = n_samples, features = n_features,
                                                                       classes = args[0], sep = sep,
                                                                       random_state = random_state)

        model.train(train_x, train_y)
        pred = model.predict(test_x)

        print("Accuracy: %.2f" % (np.sum(pred == test_y) / len(test_y) * 100) + "%")
        print("Time: %.5fms" % ((time.time() - init) * 1000))

        show(test_x, pred, test_y, model.name)
    elif isinstance(model, Regressor):
        # Regression models
        train_x, test_x, train_y, test_y = get_regression_data(samples = n_samples, features = n_features,
                                                               random_state = random_state)

        model.train(train_x, train_y)
        pred = model.predict(test_x)

        mse = np.sum((pred - test_y) ** 2)

        print("Loss:", mse / len(test_y))
        print("Time: %.5fms" % ((time.time() - init) * 1000))

        if n_features == 1:
            show_trendline(test_x, test_y, model, model.name)
    elif isinstance(model, Unsupervised):
        # Unsupervised models
        x, y = get_classification_data(samples = n_samples, features = n_features, sep = sep, supervised = False,
                                       random_state = random_state)

        pred = model.train(x)
        print("Time: %.5fms" % ((time.time() - init) * 1000))

        show(x, pred, None, model.name)


if __name__ == '__main__':
    state = 4  # Set random state for reproducibility

    name = input("Please input the model to train: ")
    train(name, random_state = state)

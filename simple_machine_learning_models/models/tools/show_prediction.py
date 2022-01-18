# Show prediction vs real label after training
# @Time: 9/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: show_prediction.py

import matplotlib.pyplot as plt
import numpy as np
from ..Regressor import Regressor

color = ["green", "blue", "yellow", "purple", "black", "pink", "red"]


def show(test_data: np.ndarray, pred, test_label: np.ndarray, model_name: str):
    assert pred is None or isinstance(pred, np.ndarray)

    outlier = [test_data[i] for i in range(len(pred)) if pred[i] < 0]
    normal = [test_data[i] for i in range(len(pred)) if pred[i] >= 0]
    label = [p for p in pred if p >= 0]

    if test_label is not None:
        plt.subplot(1, 2, 1)

    for i in range(len(label)):
        plt.scatter(normal[i][0], normal[i][1], s = 30, c = color[int(label[i])])

    for i in range(len(outlier)):
        plt.scatter(outlier[i][0], outlier[i][1], s = 30, c = "red", marker = 'x')

    plt.xlabel('X1')
    plt.ylabel('X2')

    plt.title(model_name)

    if test_label is not None:
        plt.subplot(1, 2, 2)

        for i in range(len(test_label)):
            plt.scatter(test_data[i][0], test_data[i][1], s = 30, c = color[int(test_label[i])])

        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.title("Real Labels")

    plt.show()


def show_trendline(test_data: np.ndarray, test_label: np.ndarray, model: Regressor, model_name: str):
    plt.scatter(test_data[:, 0], test_label, c = "black")
    plt.scatter(np.reshape(np.arange(-5, 5, .01), (-1, 1)),
                model.predict(np.reshape(np.arange(-5, 5, .01), (-1, 1))), c = "red", s = 1)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.title(model_name)

    plt.show()

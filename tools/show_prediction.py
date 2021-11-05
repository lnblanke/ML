# Show prediction vs real label after training
# @Time: 9/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: show_prediction.py

import matplotlib.pyplot as plt
import numpy as np

color = ["red", "green", "blue", "yellow", "purple", "black", "pink"]


def show(test_data, pred, test_label, model_name):
    plt.subplot(1, 2, 1)

    for i in range(len(test_label)):
        plt.scatter(test_data[i][0], test_data[i][1], s = 30, c = color[int(pred[i])])

    plt.xlabel('X1')
    plt.ylabel('X2')

    plt.title(model_name)

    plt.subplot(1, 2, 2)

    for i in range(len(test_label)):
        plt.scatter(test_data[i][0], test_data[i][1], s = 30, c = color[int(test_label[i])])

    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title("Real Labels")

    plt.show()


def show_trendline(test_data, test_label, model, model_name):
    plt.scatter(test_data[:, 0], test_label, c = "black")
    plt.scatter(np.reshape(np.arange(-5, 5, .01), (-1, 1)),
                model.predict(np.reshape(np.arange(-5, 5, .01), (-1, 1))), c = "red", s = 1)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.title(model_name)

    plt.show()

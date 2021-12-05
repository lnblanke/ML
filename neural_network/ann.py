# This is a sample fully connected neural network that learns to perform XOR calculation
# @Time: 8/13/2020
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: ann.py

from blocks import Dense
from blocks import mse
import numpy as np
from tools.get_data import get_classification_data
from tools.show_prediction import show

rate = 1e-2  # Learning rate
epoch = 200  # Learning epochs

if __name__ == '__main__':
    # Get data
    train_x, test_x, train_y, test_y = get_classification_data(samples = 1000, features = 2,
                                                               classes = 2, sep = 1)

    # Define model
    layers = [
        Dense(2, 64, "sigmoid", rate),
        Dense(64, 32, "sigmoid", rate),
        Dense(32, 16, "sigmoid", rate),
        Dense(16, 4, "sigmoid", rate),
        Dense(4, 1, "sigmoid", rate),
    ]

    # Train model
    prev_loss = 1

    for i in range(epoch):
        output = train_x

        for layer in layers:
            output = layer.feedforward(output)

        output = output.flatten()

        # Print learning results
        loss = mse(train_y, output)

        print("Epoch %d loss: %.3f accuracy: %.2f" % (
            i + 1, loss, np.sum((output >= .5) == train_y) / len(train_y) * 100) + "%")

        dev = output.flatten() - train_y

        for layer in reversed(layers):
            dev = layer.backprop(dev)

        if loss >= prev_loss:
            break

        prev_loss = loss

    output = test_x

    for layer in layers:
        output = layer.feedforward(output)

    output = output.flatten() > .5

    print("Accuracy: %.2f" % (np.sum(output == test_y) / len(test_y) * 100) + "%")

    show(test_x, output, test_y, "ANN")

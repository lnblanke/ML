# Testing whether a person is male or female with weight and height using common neural network
# @Time: 8/13/2020
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: ann.py

from blocks import Dense
from blocks import mse
import numpy as np

rate = 0.1  # Learning rate
epoch = 1000  # Learning epochs

# Input data
data = np.array([[-2, -1], [25, 6], [17, 4], [-15, -6]])
y = np.array([1, 0, 0, 1])

if __name__ == '__main__':
    layers = [
        Dense(2, 3, "sigmoid", rate),
        Dense(3, 5, "sigmoid", rate),
        Dense(5, 3, "sigmoid", rate),
        Dense(3, 1, "sigmoid", rate),
    ]

    # Learn
    for i in range(epoch):
        for j in range(len(y)):
            output = data[j]

            for layer in layers:
                output = layer.feedforward(output)

            # Print learning results
            if (i + 1) % 10 == 0:
                loss = mse(y, output)
                print("Epoch %d loss: %.3f" % (i + 1, loss))

            dev = -2 * (y[j] - output)

            for layer in reversed(layers):
               dev = layer.backprop(dev)
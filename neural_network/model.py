# A neural network model class that can be trained
# @Time: 12/6/21
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: model.py.py

import numpy as np
from blocks.layer import Layer
from blocks import mse, cross_entropy_loss, dmse, dcross_entropy_loss


class Model:
    def __init__(self, name = None):
        self.name = name
        self.layers = []
        self.loss = None
        self.dloss = None

    def add(self, layer: Layer):
        self.layers.append(layer)

    def _feedforward(self, train_x: np.ndarray) -> np.ndarray:
        output = train_x

        for layer in self.layers:
            output = layer.feedforward(output)

        return output

    def _backprop(self, dev: np.ndarray, learning_rate: float):
        for layer in reversed(self.layers):
            dev = layer.backprop(dev, learning_rate)

    def fit(self, train_x: np.ndarray, train_y: np.ndarray, loss_func: str, epochs: int, learning_rate = 1e-3,
            patience = 10):
        if loss_func.lower() == "mse":
            self.loss = mse
            self.dloss = dmse
        elif loss_func.lower() == "cross entropy loss":
            self.loss = cross_entropy_loss
            self.dloss = dcross_entropy_loss
        else:
            raise NameError("The loss function is not found!")

        best_loss = 1e10
        best_patience = 0

        for epoch in range(epochs):
            acc = 0
            loss = 0
            for i, (x, y) in enumerate(zip(train_x, train_y)):
                output = self._feedforward(x)

                if output.shape == (1,):
                    acc += (output >= 0.5) == y
                elif not isinstance(y, np.ndarray):
                    acc += (np.argmax(output) == y)
                else:
                    acc += np.argmax(output) == np.argmax(y)

                loss += self.loss(output, y)

                self._backprop(self.dloss(output, y), learning_rate)

            loss /= len(train_y)

            print("Epoch %d loss: %.3f accuracy: %.2f" % (
                epoch + 1, loss, acc / len(train_y) * 100) + "%")

            # A simple early stopping
            if loss < best_loss - .001:
                best_loss = loss
                best_patience = 0
            else:
                best_patience += 1

                if best_patience > patience:
                    break

    def predict(self, test_x):
        output = np.zeros(len(test_x))

        for i in range(len(test_x)):
            out = self._feedforward(test_x[i])

            if out.shape == (1,):
                output[i] = out >= .5
            else:
                output[i] = np.argmax(out)

        return output

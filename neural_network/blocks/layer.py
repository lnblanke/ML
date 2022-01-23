# Layer prototype for all implemented layers
# @Time: 12/5/21
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: layer.py.py
import abc
import numpy as np


class Layer:
    def __init__(self, name: str):
        self.name = name

    @abc.abstractmethod
    def feedforward(self, input_vector: np.ndarray):
        pass

    @abc.abstractmethod
    def backprop(self, dy_dx: np.ndarray, learning_rate):
        pass

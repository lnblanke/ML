# All the kernel functions for SVM
# @Time: 6/13/22
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: kernel.py.py

import numpy as np


def linear(x1: np.ndarray, x2: np.ndarray):
    return x1 @ x2


def poly(x1: np.ndarray, x2: np.ndarray, d = 2):
    return (x1 @ x2) ** d


def rbf(x1: np.ndarray, x2: np.ndarray, sigma = 1):
    return np.exp(-np.linalg.norm(x1 - x2, axis = -1) ** 2 / (2 * sigma ** 2))


def laplace(x1: np.ndarray, x2: np.ndarray, sigma = 1):
    return np.exp(-np.linalg.norm(x1 - x2, axis = -1) / sigma ** 2)

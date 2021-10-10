# @Time: 10/10/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: __init__.py.py

from .ada_boost import AdaBoost
from .decision_tree import DecisionTree
from .gaussian import GDA
from .k_means import KMeans
from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .naive_bayes import NaiveBayes
from .random_forest import RandomForest
from .softmax_regression import SoftmaxRegression

models = [
    AdaBoost,
    DecisionTree,
    GDA,
    KMeans,
    LinearRegression,
    LogisticRegression,
    NaiveBayes,
    RandomForest,
    SoftmaxRegression
]

supervised_models = [
    AdaBoost,
    DecisionTree,
    GDA,
    LinearRegression,
    LogisticRegression,
    NaiveBayes,
    RandomForest,
    SoftmaxRegression
]

unsupervised_models = [
    KMeans
]

classification_models = [
    AdaBoost,
    DecisionTree,
    GDA,
    LogisticRegression,
    NaiveBayes,
    RandomForest,
    SoftmaxRegression
]

regression_models = [
    LinearRegression
]
# @Time: 10/10/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: __init__.py.py

from .ada_boost import AdaBoost
from .decision_tree import DecisionTreeRegression, DecisionTreeClassification
from .gaussian import GDA
from .gradient_boost import GradBoost
from .k_means import KMeans
from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .naive_bayes import NaiveBayes
from .random_forest import RandomForest
from .softmax_regression import SoftmaxRegression

models = [
    AdaBoost,
    DecisionTreeClassification,
    DecisionTreeRegression,
    GDA,
    GradBoost,
    KMeans,
    LinearRegression,
    LogisticRegression,
    NaiveBayes,
    RandomForest,
    SoftmaxRegression
]

supervised_models = [
    AdaBoost,
    DecisionTreeClassification,
    DecisionTreeRegression,
    GDA,
    GradBoost,
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
    DecisionTreeClassification,
    GDA,
    LogisticRegression,
    NaiveBayes,
    RandomForest,
    SoftmaxRegression
]

regression_models = [
    DecisionTreeRegression,
    GradBoost,
    LinearRegression
]
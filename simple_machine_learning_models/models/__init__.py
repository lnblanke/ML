# @Time: 10/10/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: __init__.py.py

from .ada_boost import AdaBoost
from .decision_tree import DecisionTreeRegressor, DecisionTreeClassifier
from .gaussian import GDA
from .gradient_boost import GradBoost
from .k_means import KMeans
from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .naive_bayes import NaiveBayes
from .random_forest import RandomForestClassifier, RandomForestRegressor
from .softmax_regression import SoftmaxRegression

models = [
    AdaBoost,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    GDA,
    GradBoost,
    KMeans,
    LinearRegression,
    LogisticRegression,
    NaiveBayes,
    RandomForestClassifier,
    RandomForestRegressor,
    SoftmaxRegression
]

supervised_models = [
    AdaBoost,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    GDA,
    GradBoost,
    LinearRegression,
    LogisticRegression,
    NaiveBayes,
    RandomForestClassifier,
    RandomForestRegressor,
    SoftmaxRegression
]

unsupervised_models = [
    KMeans
]

classification_models = [
    AdaBoost,
    DecisionTreeClassifier,
    GDA,
    LogisticRegression,
    NaiveBayes,
    RandomForestClassifier,
    SoftmaxRegression
]

regression_models = [
    DecisionTreeRegressor,
    GradBoost,
    LinearRegression,
    RandomForestRegressor
]
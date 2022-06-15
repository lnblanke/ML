# @Time: 10/10/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: __init__.py.py

from .Model import Model
from .Supervised import Supervised
from .Unsupervised import Unsupervised
from .Classifier import Classifier
from .Regressor import Regressor
from .Ensemble import Ensemble
from .ada_boost import AdaBoost
from .dbscan import DBSCAN
from .decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
from .em import EM
from .gda import GDA
from .gradient_boost import GradBoost
from .k_means import KMeans
from .linear_regression import LinearRegression
from .logistic_regression import LogisticRegression
from .naive_bayes import NaiveBayes
from .random_forest import RandomForestClassifier, RandomForestRegressor
from .softmax_regression import SoftmaxRegression
from .stacking import StackingRegressor
from .svm import SVM
from .tools import *

models = [
    AdaBoost,
    DecisionTreeClassifier,
    DBSCAN,
    DecisionTreeRegressor,
    EM,
    GDA,
    GradBoost,
    KMeans,
    LinearRegression,
    LogisticRegression,
    NaiveBayes,
    RandomForestClassifier,
    RandomForestRegressor,
    SoftmaxRegression,
    StackingRegressor,
    SVM
]

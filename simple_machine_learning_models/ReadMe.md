# Simple Machine Learning Models

The implement of basic machine learning models

****

### Intro

In this directory, we implement some of the simplest machine learning models for both supervised and unsupervised
learning. These white-box models, like linear regression and decision tree, are easy to be understood and implemented
only with simple libraries like NumPy. The list of all the implemented models are [below](#list).

****

### Dependence

All the models in this directory are implemented only with NumPy, but some models depend on each other and external
functions implemented in this directory, explained in the [list](#list). We also use functions utilizing Matplotlib to
visualize training results and Scikit-Learn to generate random data and label for training and testing. The dependence
for these functions can be found in the [requirements file](requirements.txt). Apparently, these libraries are not
required as long as the data can be fed into the models. We also provide runnable [Jupyter Notebook](train.ipynb) that
automatically installs dependencies and builds up every model implemented here.

****

### Structure of Directory

- Simple Machine Learning Models
    - [models](models)
        - [AdaBoost](models/ada_boost)
        - [DBSCAN](models/dbscan)
        - [Decision Tree](models/decision_tree)
        - [GDA](models/gda)
        - [Gradient Boost](models/gradient_boost)
        - [K-Means](models/k_means)
        - [Linear Regression](models/linear_regression)
        - [Logistic Regression](models/logistic_regression)
        - [Naive Bayes](models/naive_bayes)
        - [Random Forest](models/random_forest)
        - [Softmax Regression](models/softmax_regression)
        - [Stacking](models/stacking)
    - [train.py](train.py) - a trainer that can be used to train each model
    - [train.ipynb](train.ipynb) - Jupyter Notebook version of the trainer

****

<h3 id = "list"> List of Implementations </h3>

| Model Type                  | Model Name                                        | Dependent Libraries                          |
|-----------------------------|---------------------------------------------------|----------------------------------------------|
| Classification              | [AdaBoost](models/ada_boost)                      | NumPy, [Decision Tree](models/decision_tree) |
| Clustering                  | [DBSCAN](models/dbscan)                           | Numpy                                        |
| Classification & Regression | [Decision Tree](models/decision_tree)             | NumPy, SciPy                                 |
| Classification              | [GDA](models/gda)                                 | NumPy                                        |
| Regression                  | [Gradient Boost](models/gradient_boost)           | NumPy, [Decision Tree](models/decision_tree) |
| Clustering                  | [K-Means](models/k_means)                         | NumPy                                        |
| Regression                  | [Linear Regression](models/linear_regression)     | NumPy                                        |
| Classification              | [Logistic Regression](models/logistic_regression) | NumPy                                        |
| Classification              | [Naive Bayes](models/naive_bayes)                 | NumPy                                        |
| Classification & Regression | [Random Forest](models/random_forest)             | NumPy, [Decision Tree](models/decision_tree) |
| Classification              | [Softmax Regression](models/softmax_regression)   | Numpy                                        |
| Regression                  | [Stacking](models/stacking)                       | Numpy, [Decision Tree](models/decision_tree) |
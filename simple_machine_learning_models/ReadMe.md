# Simple Machine Learning Models

The implement of basic machine learning models

****

### Intro

In this directory, we implement some of the simplest machine learning models for both supervised and unsupervised learning. These white-box models, like linear regression and decision tree, are easy to be understood and implemented only with simple libraries like NumPy. The list of all the implemented models are [below](#list)

****

### Dependence

All the models in this directory are implemented only with NumPy, but we import [functions](../tools/show_prediction.py) in the [tools](../tools) directory to visual visual the data and compare the prediction we make with the original labels of the data, and [functions](../tools/get_data.py) that uses function in Scikit-Learn to generate random data and label for training and test. The dependence for these functions can be found in the [README file](../tools/README.md) of the directory. Apparently, these libraries are not required as long as the data can be fed into the models.

****

### Structure of Directory

- Simple Machine Learning Models
  - [models](models)
    - [AdaBoost](models/ada_boost)
    - [Decision Tree](models/decision_tree)
    - [GDA](models/gaussian)
    - [K-Means](models/k_means)
    - [Linear Regression](models/linear_regression)
    - [Logistic Regression](models/logistic_regression)
    - [Naive Bayes](models/naive_bayes)
    - [Random Forest](models/random_forest)
    - [Softmax Regression](models/softmax_regression)

****

<h3 id = "list"> List of Implementations </h3>

| Model Type | Model Name | Dependent Libraries |
| -------- | ---------- | ----------------- | ------------------- |
| Classification | [AdaBoost](models/ada_boost) | NumPy, SciPy |
| Classification | [Decision Tree](models/decision_tree) | NumPy, [Decision Tree](models/decision_tree) |
| Classification | [GDA](models/gaussian) | NumPy |
| Unsupervised Clustering |  [K-Means](models/k_means) | NumPy |
| Regression | [Linear Regression](models/linear_regression) | NumPy, Matplotlib |
| Classification | [Logistic Regression](models/logistic_regression) | NumPy |
| Classification | [Naive Bayes](models/naive_bayes) | NumPy |
| Classification | [Random Forest](models/random_forest) | NumPy, [Decision Tree](models/decision_tree) |
| Classification | [Softmax Regression](models/softmax_regression) | Numpy |
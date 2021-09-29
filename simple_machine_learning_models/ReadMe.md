# Simple Machine Learning Models

The implement of basic machine learning models

****

### Intro

In this directory, we implement some of the simplest machine learning models for both supervised and unsupervised learning 

****

### Dependence

All the neural networks implemented in this directory are simply constructed with NumPy, but for CNN, we import MNIST library to download the data of MNIST dataset. Meanwhile, we also use Keras to create a CNN as a comparison to our model.

****

### Structure of Directory

- Neural Network
  - [ANN](ANN)
  - [CNN](CNN)
  - [RNN](RNN)

****

<h3 id = "list"> List of Implementations </h3>

| Block Name | Original Paper | Dependent Libraries |
| -------- | ---------- | ----------------- | ------------------- |
| [Artificial Neural Network](ann) | | NumPy |
| [Convolutional Neural Network](cnn) | [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) | NumPy, MNIST, (Keras) |
| [Recursive Neural Network](rnn) | [Learning Internal Representations by Error Propagation](https://apps.dtic.mil/dtic/tr/fulltext/u2/a164453.pdf) | NumPy |
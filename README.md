
# ML

Machine learning models implementation

****

### Intro

This repository contains the implementations for several machine learning(ML) and deep learning(DL) models. These models include simple machine learning models such as linear regression, logistic regression, and decision, as well as complex deep neural networks like VGG and ResNet. The detailed list of the implementations can be found [here](#list). There are also some simple, command-line ML applications in this repo. 

****

### Installation

Most implementations require dependence of NumPy, Scikit-Learn, Tensorflow, and PyTorch. Some applications depend on other libraries like OpenCV. Special requirements will be illustrated in the [list](#list). Learnt weights for deep learning models are also available in this repo.

****

### Structure of the Repository
- ML
  - [Computer Vision](cv)
  - [FaceRecog](FaceRecog)
  - [Kaggle](Kaggle)
  - [MNIST](MNIST)
  - [Neural Network](neural_network)
  - [Natural Language Processing](nlp)
  - [Simple Machine Learning Models](simple_machine_learning_models)
  - [Tools](Tools)
  - [Verification](Verification)

****

<h3 id = "list"> List of Implementations </h3>

| Category | Model Name | Implemented Paper | Dependent Libraries |
| -------- | ---------- | ----------------- | ------------------- |
| Computer Vision | [AlexNet](cv/AlexNet) | [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) | Tensorflow |
| Computer Vision | [VGG](cv/VGG.py) | [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf) | Tensorflow |
| Computer Vision | [GoogLeNet](cv/GoogLeNet.py) | [Going Deeper with Convolutions](https://arxiv.org/pdf/1409.4842.pdf) | Tensorflow |
| Computer Vision | [Xception](cv/Xception.py) | [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357.pdf) | Tensorflow |
| Computer Vision | [ResNet](cv/ResNet.py) | [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf) | Tensorflow |
| Machine Learning | [Decision Tree](simple_machine_learning_models/decision_tree) | | NumPy, Matplotlib |
| Machine Learning | [Gaussian Discriminant Analysis](simple_machine_learning_models/GDA) | | NumPy, Matplotlib |
| Machine Learning | [K-means](simple_machine_learning_models/k_means) | | NumPy, Matplotlib |
| Machine Learning | [Linear Regression](simple_machine_learning_models/linear_regression) | | NumPy, Matplotlib |
| Machine Learning | [Logistic Regression](simple_machine_learning_models/logistic_regression) | | NumPy, Matplotlib |
| Machine Learning | [Naive Bayes](simple_machine_learning_models/naive_bayes) | | NumPy, Matplotlib |
| Machine Learning | [Softmax Regression](simple_machine_learning_models/softmax_regression) | | NumPy, Matplotlib |
| Neural Network | [Artificial Neural Network](neural_network/ann) | | NumPy |
| Neural Network | [Convolutional Neural Network](neural_network/cnn) | [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) | NumPy, MNIST |
| Neural Network | [Recursive Neural Network](neural_network/rnn) | [Learning Internal Representations by Error Propagation](https://apps.dtic.mil/dtic/tr/fulltext/u2/a164453.pdf) | NumPy |
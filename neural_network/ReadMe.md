# Neural Network

The implement of fundamental neural network blocks

****

### Intro

Neural network is the basis of modern deep learning models. In this directory, we only use NumPy to create simple neural networks with feedforward and backpropagation. We implement three types of neural networks:
- Artificial Neural Network(ANN)
    Neural network only with dense layers
- Convolutional Neural Network(CNN)
    Neural network with convolutional layers and MaxPool layers, connected with one or several dense layers
- Recursive Neural Network(RNN)
    Neural network that repeats a number of times

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
| ---------- | -------------- | ------------------- |
| [Artificial Neural Network](ann) | | NumPy |
| [Convolutional Neural Network](cnn) | [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) | NumPy, MNIST, (Keras) |
| [Recursive Neural Network](rnn) | [Learning Internal Representations by Error Propagation](https://apps.dtic.mil/dtic/tr/fulltext/u2/a164453.pdf) | NumPy |

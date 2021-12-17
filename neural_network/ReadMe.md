# Neural Network

The implement of fundamental neural network blocks

****

### Intro

Neural network is the basis of modern deep learning models. In this directory, we only use NumPy to create simple neural networks with feedforward and backpropagation. We implement three types of neural networks:
- Artificial Neural Network(ANN) - Neural network only with dense layers
- Convolutional Neural Network(CNN) - Neural network with convolutional layers and MaxPool layers, connected with one or
  several dense layers
- Recursive Neural Network(RNN) - Neural network that repeats a number of times

****

### Dependence

All the neural networks implemented in this directory are simply constructed with NumPy, but for CNN, we import MNIST
library to download the data of MNIST dataset. The training of CNN is relatively slow, so we recommend using GPU to
accelerate training process.

****

### Structure of Directory

- Neural Network
    - [Blocks](blocks)
    - [ANN](ann.py)
    - [CNN](cnn.py)
    - [Multi-class ANN](multi-class-ann.py)
    - [RNN](rnn)

****

<h3 id = "list"> List of Implementations </h3>

| Block Name                            | Original Paper                                                                                                                                | Dependent Libraries |
|---------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|---------------------|
| [Convolutional layer](blocks/conv.py) | [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)                                | NumPy               |
| [Dense layer](blocks/dense.py)        | [Learning Internal Representations by Error Propagation](https://apps.dtic.mil/dtic/tr/fulltext/u2/a164453.pdf)                               | NumPy               |
| [Maxpool layer](blocks/maxpool.py)    | [Evaluation of Pooling Operations in Convolutional Architectures for Object Recognition](http://ais.uni-bonn.de/papers/icann2010_maxpool.pdf) | NumPy               |
| [Recursive layer](rnn)                | [Learning Internal Representations by Error Propagation](https://apps.dtic.mil/dtic/tr/fulltext/u2/a164453.pdf)                               | NumPy               |

We also implemented a [generalized model class](model.py) that is similar to tf.keras.Model and used these layers to
perform several tasks, including [binary classification](ann.py), [multi-class classification](multi-class-ann.py),
and [MNIST classification](cnn.py).
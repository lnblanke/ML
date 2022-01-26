# Neural Network

The implement of fundamental neural network blocks

****

### Intro

Neural network is the basis of modern deep learning models. In this directory, we only use NumPy to create simple neural
networks with feedforward and backpropagation. We implement three types of neural networks:

- Artificial Neural Network(ANN) - Neural network only with dense layers
- Convolutional Neural Network(CNN) - Neural network with convolutional layers and MaxPool layers, connected with one or
  several dense layers
- Recursive Neural Network(RNN) - Neural network that repeats a number of times

****

### Dependence

All the neural networks implemented in this directory are simply constructed with NumPy, but for CNN and RNN, we import
TensorFlow library to download the data of MNIST dataset and IMDB dataset. You can also install MNIST and IMDB libraries
to get data in replace of TensorFlow. The training of CNN and RNN is relatively slow, so we recommend using GPU to
accelerate training process.

We also use pretrained embedding layer from [Keras](https://keras.io/examples/nlp/pretrained_word_embeddings/) for
preprocessing the IMDB data for RNN training. You can download it using the following commands:

```shell
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip -q glove.6B.zip
```

****

### Structure of Directory

- Neural Network
  - [Blocks](blocks)
  - [ANN](ann.py)
  - [CNN](cnn.py)
  - [Multi-class ANN](multi-class-ann.py)
    - [RNN](rnn.py)

****

<h3 id = "list"> List of Implementations </h3>

| Block Name                            | Original Paper                                                                                                                                |
|---------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| [Convolutional layer](blocks/conv.py) | [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)                                |
| [Dense layer](blocks/dense.py)        | [Learning Internal Representations by Error Propagation](https://apps.dtic.mil/dtic/tr/fulltext/u2/a164453.pdf)                               |
| [Maxpool layer](blocks/maxpool.py)    | [Evaluation of Pooling Operations in Convolutional Architectures for Object Recognition](http://ais.uni-bonn.de/papers/icann2010_maxpool.pdf) |
| [Recursive layer](blocks/rnn.py)      | [Learning Internal Representations by Error Propagation](https://apps.dtic.mil/dtic/tr/fulltext/u2/a164453.pdf)                               |

We also implemented a [generalized model class](model.py) that is similar to ```tf.keras.Model``` and used these layers
to perform several tasks, including [binary classification](ann.py), [IMDB dataset classification](rnn.py),
and [MNIST classification](cnn.py).

****

### Reference

The backpropagation process of dense layer is inspired by Michael Nelson's
book [Neural Network and Deep Learning, Chap. 4](http://neuralnetworksanddeeplearning.com/index.html). The
backpropagation of convolutional and recursive layers are adapted from Victor Zhou's
blogs [Neural Networks from Scratch](https://victorzhou.com/series/neural-networks-from-scratch).
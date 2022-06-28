# AlexNet

![paper.png](paper.png)

## Intro

This directory is the implementation of AlexNet, the convolutional neural network structure demonstrated in the
paper [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)
by Krizhevsky et al. published on NIPS 2012. In this paper, the authors utilized a nine-layer neural network containing
convolutional layers, pooling layers, and fully connected layers to predict ImageNet images. The detailed description of
the model can be found in the paper listed above. In this directory, we used TensorFlow and Keras framework to implement
a simple AlexNet model and trained our fine-tuned model with Cifar 10 model, receiving a top-1 accuracy of 84% on test set using data augmentation and dropout.

### Structure of Directory

- [AlexNet.py](AlexNet.py)
  This python file contains a function that builds a raw AlexNet model that can be trained on ImageNet dataset.
- [AlexNet.png](AlexNet.png)
  This png file shows the structure of AlexNet that can be trained on ImageNet dataset.
- [AlexNet.ipynb](AlexNet.ipynb)
  This Jupyter Notebook records our training of AlexNet on Cifar 10 dataset. It can be run on any devices.
- [Weights](weights)
  This directory contains the weights that we trained on Cifar 10 dataset.
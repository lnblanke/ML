# VGG

## Intro

This directory is the implementation of VGG, the convolutional neural network structure demonstrated in the paper [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf) by Simonyan et al. published on ICLR 2015. In this paper, the authors utilized an up-to-19-layer neural network containing convolutional layers, pooling layers, and fully connected layers to predict ImageNet images. The detailed description of the model can be found in the paper listed above. In this directory, we used TensorFlow and Keras framework to implement a simple VGG model and trained our fine-tuned model with Cifar 10 model, receiving a top-1 accuracy of 87%.

### Structure of Directory

- [VGG.py](VGG.py)
    This python file contains a function that builds an VGG model that can be trained on ImageNet dataset.
- [VGG.png](VGG.png)
    This png file shows the structure of VGG that can be trained on ImageNet dataset.
- [VGG.ipynb](VGG.ipynb)
    This Jupyter Notebook records our training of VGG on Cifar 10 dataset. It can be run on any devices
- [VGG.ipynb](VGG.ipynb)
    This Jupyter Notebook is the original file that we used to train the VGG on Google Colab. It can be directly run on Colab.
- [Weights](weights)
    This directory contains the weights that we trained on Cifar 10 dataset.
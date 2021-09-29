# Computer Vision

The implement of computer vision models in recent years

****

### Intro

Computer vision(CV) is one of the major areas of deep learning nowadays. It mainly deals with segmentation and recognition of objects in picture. Some of the fields of CV are image classification, object detection, and semantic segmentation. This directory of the repo implements some of the most important CV models in recently years, including AlexNet, VGG, GoogLeNet, and ResNet. Detailed list of CV models in this directory can be found [here](#list).

****

### Dependence

All the CV models in this directory are constructed based on TensorFlow 2. The data of Cifar 10 dataset is acquired from Tensorflow Dataset and we use Keras-Tuner to fine-tune our models. For Jyputer Notebooks, all the dependence libraries can be installed simply by running the cells. GPU is preferred for fine-tuning and training process as deep neural networks contains millions of weights that are hard to train in a very short time.

****

### Structure of Directory

- Computer Vision
  - [AlexNet](AlexNet)
  - [VGG](VGG.py)
  - [GoogLeNet](GoogLeNet.py)
  - [Xception](Xception.py)
  - [ResNet](ResNet.py)

****

<h3 id = "list"> List of Implementations </h3>

| Category | Model Name | Implemented Paper | Dependent Libraries |
| -------- | ---------- | ----------------- | ------------------- |
| Image Classification | [AlexNet](AlexNet) | [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) | Tensorflow |
| Image Classification | [VGG](VGG.py) | [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf) | Tensorflow |
| Image Classification | [GoogLeNet](GoogLeNet.py) | [Going Deeper with Convolutions](https://arxiv.org/pdf/1409.4842.pdf) | Tensorflow |
| Image Classification | [Xception](Xception.py) | [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357.pdf) | Tensorflow |
| Image Classification | [ResNet](ResNet.py) | [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/pdf/1409.1556.pdf) | Tensorflow |

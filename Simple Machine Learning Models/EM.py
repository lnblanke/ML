# A simple EM unsupervised classification model
# @Time: 6/20/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: EM.py

import numpy as np
import time
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

data = make_classification ( n_samples = 1000 , n_classes = 2 , n_clusters_per_class = 1 , n_features = 2 , n_redundant = 0 , class_sep = 2 )

train_data = data [ 0 ] [ : 1000 ]

if __name__ == '__main__' :
    init = time.time ()
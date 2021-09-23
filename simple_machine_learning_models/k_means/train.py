# Train a K-means classifier
# @Time: 9/6/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: train.py

from tools import *
import time
from kmeans import kmeans

if __name__ == '__main__':
    init = time.time()

    data, label = get_classification_data(sep = 4, supervised = False)

    k = kmeans(2)
    pred = k.train(data)

    print("Time: %.5fms" % ((time.time() - init) * 1000))

    show(data, pred, label, "K-means")

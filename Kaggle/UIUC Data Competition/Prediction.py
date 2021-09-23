# Prediction for the competition
# @Time: 9/17/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Prediction.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *

train = pd.read_csv("train_30_PHYDX_3.csv")
test = pd.read_csv("test_30_PHYDX_3.csv")

weight = np.random.rand(7)

model = tf.keras.Sequential([

])

plt.scatter(train.x, train.R, c = "red", s = 2)
plt.scatter(np.arange(-2, 2, 0.001), model.predict(np.arange(-2, 2, .001)), c = "blue", s = 2)
plt.show()

pred = model.predict(test.x)

output = pd.DataFrame({"X": test.x, "R": pred[:, 0]})
output.to_csv("submission.csv", index = False)

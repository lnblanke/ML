# @Time: 3/12/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Prediction.py

# Train a CNN with Keras and make predictions
# @Time: 8/14/2020
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Keras.py

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D , Dense , Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
import pandas as pd
import matplotlib.pyplot as plt
import pylab

train = pd.read_csv ( "train.csv" )
test = pd.read_csv ( "test.csv" )

train_label = train [ "label" ].values

del train [ "label" ]

train_img = train.values
test_img = test.values

train_img = np.reshape ( train_img / 255 - .5 , ( len ( train_img ) , 28 , 28 ) )
test_img = np.reshape ( test_img / 255 - .5 , ( len ( test_img ) , 28 , 28 ) )

train_img = train_img
train_label = train_label

train_img = np.expand_dims ( train_img , 3 )
test_img = np.expand_dims ( test_img , 3 )

# Form the CNN model
model = Sequential ( [
    Conv2D ( 8 , 3 , input_shape = (28 , 28 , 1) , use_bias = False ) ,
    MaxPooling2D ( pool_size = 2 ) ,
    Flatten () ,
    Dense ( 10 , activation = 'softmax' )
] )

# Train
model.compile ( SGD ( lr = .005 ) , loss = 'categorical_crossentropy' , metrics = [ 'accuracy' ] )

model.fit (
    train_img ,
    to_categorical ( train_label ) ,
    batch_size = 1 ,
    epochs = 20 ,
)

# Test
prediction = model.predict ( test_img )

id = np.array ( len ( test_img ) )

df = pd.DataFrame ( { "Label" : np.argmax ( prediction , axis = 1 ) } , index = list ( range ( 1 , len ( test_img ) + 1 ) ) )
df.to_csv ( "prediction.csv" )
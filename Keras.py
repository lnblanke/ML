# Train a CNN with Keras and make predictions
# @Time: 8/14/2020
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Keras.py

import numpy as np
import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D , Dense , Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

train_img = mnist.train_images () [ :10000 ]
train_label = mnist.train_labels () [ :10000 ]
test_img = mnist.test_images () [ :1000 ]
test_label = mnist.test_labels () [ :1000 ]

train_img = train_img / 255 - 0.5
test_img = test_img / 255 - 0.5

train_img = np.expand_dims ( train_img , 3 )
test_img = np.expand_dims ( test_img , 3 )

model = Sequential ( [
    Conv2D ( 8 , 3 , input_shape = (28 , 28 , 1) , use_bias = False ) ,
    MaxPooling2D ( pool_size = 2 ) ,
    Flatten () ,
    Dense ( 10 , activation = 'softmax' )
] )

model.compile ( SGD ( lr = .005 ) , loss = 'categorical_crossentropy' , metrics = [ 'accuracy' ] )

model.fit (
    train_img ,
    to_categorical ( train_label ) ,
    batch_size = 1 ,
    epochs = 3 ,
    validation_data = (test_img , to_categorical ( test_label )) ,
)

model.save_weights ( "CNN_weights.h5" )

prediction = model.predict ( test_img )

print ( np.argmax ( prediction , axis = 1 ) )
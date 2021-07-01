# Train a CNN with Keras and make predictions
# @Time: 8/14/2020
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Keras.py

import numpy as np
from tensorflow.python.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D , MaxPooling2D , Dense , Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

( train_img , train_label ) , ( test_img , test_label ) = mnist.load_data ()

train_img = np.reshape ( train_img [ :1000 ] / 255 - .5 , ( 1000 , 28 , 28 ) )
test_img = np.reshape ( test_img [ :100 ] / 255 - .5 , ( 100 , 28 , 28 ) )

train_label = train_label [ :1000 ]
test_label = test_label [ :100 ]

train_img = np.expand_dims ( train_img , 3 )
test_img = np.expand_dims ( test_img , 3 )

# Form the CNN model
model = Sequential ( [
    Conv2D ( 8 , 3 , input_shape = (28 , 28 , 1) , use_bias = False ) ,
    MaxPooling2D ( pool_size = 2 ) ,
    Flatten () ,
    Dense ( 10 , activation = 'softmax' )
] )

if __name__ == '__main__':
    # Train
    model.compile ( SGD ( lr = .005 ) , loss = 'categorical_crossentropy' , metrics = [ 'accuracy' ] )

    model.fit (
        train_img ,
        to_categorical ( train_label ) ,
        batch_size = 1 ,
        epochs = 3 ,
    )

    # Test
    prediction = model.predict ( test_img )

    print ( np.argmax ( prediction , axis = 1 ) )
    print ( test_label )
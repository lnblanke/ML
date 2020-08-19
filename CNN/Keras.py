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
from PIL import Image
import os

arr = [ ]

count = 0

train_label = [ ]

for _ , _ , files in os.walk ( "../Training" ) :
    for file in files :
        for k in range ( 100 ) :
            train_img = Image.open ( os.path.join ( "../Training" , file ) ).convert ( "L" )

            train_label.append ( file [ : 1 ] )

            if train_img.size [ 0 ] != 28 or train_img.size [ 1 ] != 28 :
                train_img.resize ( (28 , 28) )

            for i in range ( 28 ) :
                for j in range ( 28 ) :
                    pixel = 255 - int ( train_img.getpixel ( (j , i) ) )

                    arr.append ( pixel )

            count += 1

train_img = np.array ( arr ).reshape ( (count , 28 , 28) )

arr = [ ]

count = 0

for _ , _ , files in os.walk ( "../Testset" ) :
    for file in files :
        test_img = Image.open ( os.path.join ( "../Testset" , file ) ).convert ( "L" )

        if test_img.size [ 0 ] != 28 or test_img.size [ 1 ] != 28 :
            test_img.resize ( (28 , 28) )

        for i in range ( 28 ) :
            for j in range ( 28 ) :
                pixel = 255 - int ( test_img.getpixel ( (j , i) ) )

                arr.append ( pixel )

        count += 1

test_img = np.array ( arr ).reshape ( (count , 28 , 28) )

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
    epochs = 1 ,
)

prediction = model.predict ( test_img )

print ( np.argmax ( prediction , axis = 1 ) )

for _ , _ , files in os.walk ( "../Testset" ) :
    print ( files )

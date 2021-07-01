# A simple VGG model on ImageNet classification using tensorflow
# @Time: 6/25/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: VGG.py

import tensorflow as tf
from tensorflow.keras.layers import *

def add_layer ( model , num_filters , num_layers ) :
    for i in range ( num_layers ) :
        model.add ( Conv2D ( filters = num_filters , kernel_size = ( 3 , 3 ) , activation = "relu" , padding = "same" ) )

    model.add ( BatchNormalization () )
    model.add ( MaxPooling2D ( pool_size = 2 , strides = 2 ) )

def createVGG ( type ) :
    if type == 11 :
        params = [ 1 , 1 , 2 , 2 , 2 ]
    elif type == 13 :
        params = [ 2 , 2 , 2 , 2 , 2 ]
    elif type == 16 :
        params = [ 2 , 2 , 3 , 3 , 3 ]
    elif type == 19 :
        params = [ 2 , 2 , 4 , 4 , 4 ]
    else :
        raise Exception ( "The parameter is not valid!" )

    model = tf.keras.Sequential ()

    model.add ( InputLayer ( input_shape = ( 224 , 224 , 3 ) ) )
    add_layer ( model , 64 , params [ 0 ] )
    add_layer ( model , 128 , params [ 1 ] )
    add_layer ( model , 256 , params [ 2 ] )
    add_layer ( model , 512 , params [ 3 ] )
    add_layer ( model , 512 , params [ 4 ] )

    model.add ( Dropout ( rate = .5 ) )
    model.add ( Activation ( activation = "relu" ) )
    model.add ( Dropout ( rate = .5 ) )
    model.add ( Activation ( activation = "relu" ) )
    model.add ( Dense ( 1000 , activation = "sigmoid" ) )

    return model

if __name__ == '__main__':
    print ( createVGG ( 19 ).summary () )
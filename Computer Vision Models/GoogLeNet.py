# A simple GoogLeNet on ImageNet classification using tensorflow
# @Time: 6/25/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: GoogLeNet.py

import tensorflow as tf
from tensorflow.keras.layers import *

def inception ( input , f1 , f3r , f3 , f5r , f5 , pp , output = False ) :
    x1 = Conv2D ( kernel_size = 1 , filters = f1 , activation = "relu" , padding = "same" ) ( input )

    x2 = Conv2D ( kernel_size = 1 , filters = f3r , activation = "relu" , padding = "same" ) ( input )
    x2 = Conv2D ( kernel_size = 3 , filters = f3 , activation = "relu" , padding = "same" ) ( x2 )

    x3 = Conv2D ( kernel_size = 1 , filters = f5r , activation = "relu" , padding = "same" ) ( input )
    x3 = Conv2D ( kernel_size = 5 , filters = f5 , activation = "relu" , padding = "same" ) ( x3 )

    x4 = MaxPooling2D ( pool_size = 3 , padding = "same" , strides = 1 ) ( input )
    x4 = Conv2D ( kernel_size = 1 , filters = pp , activation = "relu" , padding = "same" ) ( x4 )

    x = Concatenate () ( [ x1 , x2 , x3 , x4 ] )
    x = BatchNormalization () ( x )

    if output == False :
        return x , None
    else :
        x5 = AveragePooling2D ( pool_size = 5 , padding = "valid" , strides = 3 ) ( x )
        x5 = Conv2D ( kernel_size = 1 , padding = "same" , filters = 128 , activation = "relu" ) ( x5 )
        x5 = Dense ( 1024 , activation = "relu" )  ( x5 )
        x5 = Dropout ( rate = .7 ) ( x5 )
        x5 = Dense ( 1000 , activation = "softmax" ) ( x5 )

        return x , x5

def createGoogLeNet () :
    input = Input ( shape = ( 224 , 224 , 3 ) )
    x = Conv2D ( kernel_size = 7 , strides = 2 , filters = 64 , padding = "same" , activation = "relu" ) ( input )
    x = MaxPooling2D ( pool_size = 3 , strides = 2 , padding = "same" ) ( x )
    x = BatchNormalization () ( x )
    x = Conv2D ( kernel_size = 1 , strides = 1 , filters = 192 , padding = "valid" , activation = "relu" ) ( x )
    x = Conv2D ( kernel_size = 3 , strides = 1 , filters = 192 , padding = "same" , activation = "relu" ) ( x )
    x = BatchNormalization () ( x )
    x = MaxPooling2D ( pool_size = 3 , strides = 2 , padding = "same" ) ( x )
    x , _ = inception ( x , 64 , 96 , 128 , 16 , 32 , 32 )
    x , _ = inception ( x , 128 , 128 , 192 , 32 , 96 , 64 )
    x = MaxPooling2D ( pool_size = 3 , strides = 2 , padding = "same" ) ( x )
    x , _ = inception ( x , 192 , 96 , 208 , 16 , 48 , 64 )
    x , softmax_0 = inception ( x , 160 , 112 , 224 , 24 , 74 , 64 , output = True )
    x , _ = inception ( x , 128 , 128 , 256 , 24 , 64 , 64 )
    x , _ = inception ( x , 112 , 144 , 288 , 32 , 64 , 64 )
    x , softmax_1 = inception ( x , 256 , 160 , 320 , 32 , 128 , 128 , output = True )
    x = MaxPooling2D ( pool_size = 3 , strides = 2 , padding = "same" ) ( x )
    x , _ = inception ( x , 256 , 160 , 320 , 32 , 128 , 128 )
    x , _ = inception ( x , 384 , 192 , 384 , 48 , 128 , 128 )
    x = AveragePooling2D ( pool_size = 7 , padding = "valid" ) ( x )
    x = Dropout ( rate = .4 ) ( x )
    softmax_2 = Dense ( 1000 , activation = "softmax" ) ( x )

    model = tf.keras.Model ( inputs = input , outputs = [ softmax_0 , softmax_1 , softmax_2 ] , name = "GoogLeNet" )

    return model

if __name__ == '__main__' :
    print ( createGoogLeNet ().summary () )
    tf.keras.utils.plot_model ( createGoogLeNet () , to_file = "GoogLeNet.png" , show_shapes = False , show_layer_names = True , rankdir = "TB" )
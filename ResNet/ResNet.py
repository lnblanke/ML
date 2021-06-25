# A simple ResNet 34
# @Time: 6/21/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: ResNet.py

import tensorflow as tf
from Block import Block

class ResNet ( tf.keras.Model ) :
    def __init__ ( self ) :
        super ( ResNet , self ).__init__ ()

    def call ( self , inputs , training = None , mask = None ) :
        result = tf.keras.layers.Conv2D ( input , kernel_size = ( 7 ,  7 ) , strides = 2 , filters = 64 )
        result = tf.nn.batch_normalization ( result )
        result = tf.keras.layers.MaxPooling2D ( result , strides = 2 )

        layers = tf.keras.Sequential ()

        for i in range ( 3 ) :
            layers.add ( Block ( filter = 64 , stride = 1 ) )

        layers.add ( Block ( filter = 128 , stride = 2 ) )

        for i in range ( 3 ) :
            layers.add ( Block ( filter = 128 , stride = 1 ) )

        layers.add ( Block ( filter = 256 , stride = 2 ) )

        for i in range ( 5 ) :
            layers.add ( Block ( filter = 256 , stride = 1 ) )

        layers.add ( Block ( filter = 512 , stride = 2 ) )

        for i in range ( 2 ) :
            layers.add ( Block ( filter = 512 , stride = 1 ) )

        result = layers ( result )
        result = tf.nn.avg_pool2d ( result )
        result = tf.keras.layers.Dense ( result , activation = "relu" )

        return result

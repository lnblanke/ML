# A single block for ResNet
# @Time: 6/21/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Block.py

import tensorflow as tf

class Block ( tf.keras.layers.Layer ) :
    def __init__ ( self , filter , stride ) :
        super ( Block , self ).__init__ ()
        self.filters = filter
        self.strides = stride

    def call ( self , inputs , **kwargs ) :
        result = tf.keras.layers.Conv2D ( inputs , kernel_size = ( 3 , 3 ) , filters = self.filters , padding = "SAME" )
        result = tf.nn.batch_normalization ( result )
        result = tf.nn.relu ( result )
        result = tf.keras.layers.Conv2D ( result , kernel_size = ( 3 , 3 ) , filters = self.filters , padding = "SAME" )
        result = tf.nn.batch_normalization ( result )
        result = tf.keras.layers.add ( [ result , inputs ] )
        result = tf.nn.relu ( result )

        return result
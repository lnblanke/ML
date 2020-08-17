# @Time: 8/14/2020
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: RNN_class.py

import numpy as np

class recur :
    def __init__ ( self , input_size , output_size , hidden_layer = 64 ) :
        self.Whh = np.random.randn ( hidden_layer , hidden_layer ) / 1000
        self.Wxh = np.random.randn ( hidden_layer , input_size ) / 1000
        self.Why = np.random.randn ( output_size , hidden_layer ) / 1000

        self.Bh = np.zeros ( ( hidden_layer , 1 ) )
        self.By = np.zeros ( ( output_size , 1 ) )

    def forward ( self , inputs ) :
        h = np.zeros ( ( self.Whh.shape [ 0 ] , 1 ) )

        for i , x  in enumerate ( inputs ) :
            h = np.tanh ( self.Wxh @ x + self.Whh @ h + self.Bh )

        y = self.Why @ h + self.By

        return y , h 
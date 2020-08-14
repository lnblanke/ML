# Class for CNN
# @Time: 8/13/2020
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: CNNclass.py

import numpy as np

class Conv :
    def __init__ ( self , num ) :
        # Number of layers in CNN
        self.num = num

        # Initialize filters
        self.filters = np.random.randn ( num , 3 , 3 ) / 9

    # Find all 3x3 regions to be iterated
    def iter_region ( self , img ) :
        h , w = img.shape

        for i in range ( h - 2 ) :
            for j in range ( w - 2 ) :
                region = img [ i : i + 3 , j : j + 3 ]

                yield region , i , j

    # Get pic through each layers
    def forward ( self , input ) :
        h , w = input.shape

        self.input = input

        output = np.zeros ( ( h - 2 , w - 2 , self.num ) )

        for region , i , j in self.iter_region ( input ) :
            output [ i ] [ j ] = np.sum ( region * self.filters , axis = ( 1 , 2 ) )

        return output

    def backprop ( self , dL_dout , rate ) :
        dL_dfilter = np.zeros ( self.filters.shape )

        for region , i , j in self.iter_region ( self.input ) :
            for k in range ( self.num ) :
                dL_dfilter [ k ] += dL_dout [ i , j , k ] * region

        self.filters -= rate * dL_dfilter

        return None
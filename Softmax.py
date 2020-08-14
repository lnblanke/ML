# Do softmax operation
# @Time: 8/14/2020
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Softmax.py
import numpy as np

class Softmax :
    def __init__ ( self , length , node ) :
        self.weight = np.random.randn ( length , node ) / length
        self.bias = np.zeros ( node )

    def forward ( self , input ) :
        self.shape = input.shape

        input = input.flatten ()
        self.input = input

        length , node = self.weight.shape

        tot = np.dot ( input , self.weight ) + self.bias
        self.total = tot

        exp = np.exp ( tot )

        return exp / np.sum ( exp , axis = 0 )

    def backprop ( self , dL_douts , rate ) :
        for i , grad in enumerate ( dL_douts ) :
            if grad == 0 :
                continue

            e_tot = np.exp ( self.total )

            S = np.sum ( e_tot )

            douts_dt = - e_tot * e_tot [ i ] / S ** 2
            douts_dt [ i ] = e_tot [ i ] * ( S - e_tot [ i ] ) / S ** 2

            dt_dw = self.input
            dt_db = 1
            dt_dinput = self.weight
            dL_dt = grad * douts_dt

            dL_dw = dt_dw [ np.newaxis ].T @ dL_dt [ np.newaxis ]
            dL_db = dL_dt * dt_db
            dL_dinput = dt_dinput @ dL_dt

            self.weight -= rate * dL_dw
            self.bias -= rate * dL_db
            return dL_dinput.reshape ( self.shape )
# @Time: 8/14/2020
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: RNN.py

from data import train_data , test_data
import numpy as np

vocab = list ( set ( [ w for text in train_data.keys () for w in text.split ( ' ' ) ] ) )
size = len ( vocab )

print ( vocab )

word_index = { w: i for i , w in enumerate ( vocab ) }
index_word = { i: w for i , w in enumerate ( vocab ) }

def create_input ( text ) :
    inputs = []

    for word in text.split ( " " ) :
        v = np.zeros ( ( size , 1 ) )

        v [ word_index [ w ] ] = 1

        inputs.append ( v )

    return inputs


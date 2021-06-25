# @Time: 8/14/2020
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: rnn.py

from data import train_data , test_data
import numpy as np
from rnn_class import recur
import random

vocab = list ( set ( [ w for text in train_data.keys () for w in text.split ( ' ' ) ] ) )
size = len ( vocab )

word_index = { w : i for i , w in enumerate ( vocab ) }
index_word = { i : w for i , w in enumerate ( vocab ) }


def process_data ( data , backprop = True ) :
    items = list ( data.items () )
    random.shuffle ( items )

    loss = 0
    correct = 0

    for x , y in items :
        inputs = create_input ( x )
        target = int ( y )

        out , _ = rnn.forward ( inputs )
        probs = softmax ( out )

        loss -= np.log ( probs [ target ] )
        correct += int ( np.argmax ( probs ) == target )

        if backprop :
            dL_dy = probs
            dL_dy [ target ] -= 1

            rnn.backprop ( dL_dy )

    return loss / len ( data ) , correct / len ( data )


def create_input ( text ) :
    inputs = []

    for word in text.split ( " " ) :
        v = np.zeros ( ( size , 1 ) )

        v [ word_index [ word ] ] = 1

        inputs.append ( v )

    return inputs


def softmax ( x ) :
    return np.exp ( x ) / sum ( np.exp ( x ) )

rnn = recur ( size , 2 )

for epoch in range ( 1000 ) :
    ls , corr = process_data ( train_data )

    if (epoch + 1) % 100 == 0 :
        print ( '--- Epoch %d' % (epoch + 1) )
        print ( 'Train:\tLoss %.3f | Accuracy: %.3f' % (ls , corr) )

        test_loss , test_acc = process_data ( test_data , backprop = False )
        print ( 'Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss , test_acc) )

inputs = create_input ( "i am very good" )

out , _ = rnn.forward ( inputs )

print ( softmax ( out ) )
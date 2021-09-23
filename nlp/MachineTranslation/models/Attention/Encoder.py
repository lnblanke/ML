# An encoder layer
# @Time: 8/2/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Encoder.py

import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    def __init__(self, input_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(input_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(enc_units, return_sequences = True, return_state = True,
            recurrent_initializer = "glorot_uniform")

    def call(self, inputs, state = None):
        vectors = self.embedding(inputs)
        output, state = self.gru(vectors, initial_state = state)

        return output, state

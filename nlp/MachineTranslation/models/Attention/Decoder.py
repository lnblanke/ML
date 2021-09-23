# A decoder layer
# @Time: 8/2/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Decoder.py
import typing

import tensorflow as tf
from Attention import Attention
from typing import Tuple, Any

class DecoderInput(typing.NamedTuple):
    new_tokens: Any
    enc_output: Any
    mask: any

class DecoderOutput(typing.NamedTuple):
    logits: Any
    attention_weights: Any

class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(output_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(dec_units, return_state = True, return_sequences = True,
            recurrent_initializer = "glorot_uniform")
        self.attention = Attention(dec_units)
        self.D1 = tf.keras.layers.Dense(dec_units, activation = tf.math.tanh, use_bias = False)
        self.D2 = tf.keras.layers.Dense(output_size)

def call(self, inputs: DecoderInput, state = None) -> Tuple[DecoderInput, tf.Tensor]:
    vectors = self.embedding(inputs.new_tokens)
    rnn_output, state = self.gru(vectors, initial_state = state)
    context_vector, attention_weights = self.attention(query = rnn_output, value = inputs.enc_output,
        mask = inputs.mask)
    context_and_rnn_output = tf.concat([context_vector, rnn_output], axis = 1)
    attention_vector = self.D1(context_and_rnn_output)
    logits = self.D2(attention_vector)

    return DecoderOutput(logits, attention_weights), state

Decoder.call = call

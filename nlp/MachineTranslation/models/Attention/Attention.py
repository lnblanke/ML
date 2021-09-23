# An attention layer
# @Time: 8/2/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Attention.py

import tensorflow as tf

class Attention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(Attention, self).__init__()

        self.Dense1 = tf.keras.layers.Dense(units, use_bias = False)
        self.Dense2 = tf.keras.layers.Dense(units, use_bias = False)

        self.attention = tf.keras.layers.AdditiveAttention()

    def call(self, query, value, mask):
        d1_query = self.Dense1(query)
        d2_key = self.Dense2(value)
        query_mask = tf.ones(tf.shape(query)[: -1], dtype = bool)
        value_mask = mask

        context_vector, attention_weights = self.attention(inputs = [d1_query, value, d2_key],
            mask = [query_mask, value_mask], return_attention_scores = True)

        return context_vector, attention_weights

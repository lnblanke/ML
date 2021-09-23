# Set up the training model
# @Time: 8/2/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: TranslationTrain.py

import tensorflow as tf
from Encoder import Encoder
from Decoder import Decoder

class TranslationTrain(tf.keras.models.Model):
    def __init__(self, embedding_dim, units, input_processor, output_processor, use_tf_function = False):
        super().__init__()

        self.encoder = Encoder(input_processor.vocabulary_size(), embedding_dim, units)
        self.decoder = Decoder(output_processor.vocabulary_size(), embedding_dim, units)

        self.use_tf_function = use_tf_function
        self.input_processor = input_processor
        self.output_processor = output_processor

    def train_step(self, data):
        if self.use_tf_function:
            return self._tf_train_step(data)
        else:
            return self._train_step(data)

def _preprocess(self, input, target):
    input_token = self.input_processor(input)
    target_token = self.output_processor(target)

    input_mask = input_token != 0
    target_mask = target_token != 0

    return input_token, input_mask, target_token, target_mask

def _train_step(self, input):
    input_text, target_text = input

    input_token, input_mask, target_token, target_mask = self._preprocessor(input_text, target_text)

    with tf.GradientTape as tape:
        enc_out, enc_state = self.encoder(input_token)
        dec_state = enc_state
        loss = tf.constant(0.0)

TranslationTrain._preprocess = _preprocess

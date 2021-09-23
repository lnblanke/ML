# @Time: 8/3/2021
# @Author: lnblanke
# @Email: fjh314.84@gmail.com
# @File: Encoder-Decoder.py

from load_data import load_data
import tensorflow as tf
import tensorflow_text as tft

batch_size = 64
max_vocab_size = 50000
embedding_dim = 256
units = 1024

def text_norm(text):
    text = tf.strings.lower((tft.normalize_utf8(text, "NFKD")))
    text = tf.strings.regex_replace(text, "[.?!,¿:]", r" \0 ")
    text = tf.strings.strip(text)
    text = tf.strings.join(["<SOS>", text, "<EOS>"], separator = " ")

    return text

if __name__ == '__main__':
    en_train, zh_train, en_dev, zh_dev, en_test, zh_test = load_data()

    dataset = tf.data.Dataset.from_tensor_slices((en_train, zh_train)).shuffle(len(en_train)).batch(batch_size)

    input_preprocess = tf.keras.layers.experimental.preprocessing.TextVectorization(standardize = text_norm,
        max_tokens = max_vocab_size)
    input_preprocess.adapt(en_train)

    output_preprocess = tf.keras.layers.experimental.preprocessing.TextVectorization(standardize = text_norm,
        max_tokens = max_vocab_size)
    output_preprocess.adapt(zh_train)

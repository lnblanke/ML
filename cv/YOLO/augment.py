import tensorflow as tf

class Augment(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def call(self, x):
        return self.augment(x)
    
    def augment(self, x):
        seed = tf.random.uniform(shape=[2], maxval=100, dtype=tf.int32, seed=0)
        x = tf.keras.layers.Rescaling(1. / 255)(x)
        x = tf.image.stateless_random_brightness(x, .2, seed)
        x = tf.image.stateless_random_contrast(x, .2, .5, seed)
        x = tf.image.stateless_random_hue(x, .2, seed)
        x = tf.image.stateless_random_saturation(x, .2, .5, seed)
        
        return x
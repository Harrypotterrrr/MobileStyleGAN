import tensorflow as tf

from tensorflow import nn
import tensorflow.keras as keras

class NoiseInjection(keras.Model):

    def __init__(self):
        super(NoiseInjection, self).__init__()
        # TODO weight zeros used for?
        self.weight = tf.Variable(tf.zeros(1))
        self.trace_model = False

    def call(self, image, noise=None):
        if noise is None:
            # batch, _, height, width = image.shape
            # noise = image.new_empty(batch, 1, height, width).normal_()
            noise = tf.random.normal(image.shape)
        # TODO register_buffer
        # if not hasattr(self, "noise") and self.trace_model:
        #     self.register_buffer("noise", noise)
        if self.trace_model:
            noise = self.noise
        return image + self.weight * noise
import tensorflow as tf
import tensorflow.keras as keras

from models.modules.utils import ModulatedConv2d

class MultichannelIamge(keras.Model):

    def __init__(
        self,
        channels_in,
        channels_out,
        style_dim,
        kernel_size=1
    ):
        super(MultichannelIamge, self).__init__()
        self.conv = ModulatedConv2d(channels_in, channels_out, style_dim, kernel_size, demodulate=False)
        self.bias = tf.Variable(tf.zeros([1, channels_out, 1, 1]))

    def call(self, hidden, style):
        out = self.conv(hidden, style)
        out = out + self.bias
        return out
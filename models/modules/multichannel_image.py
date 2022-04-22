import tensorflow as tf
import tensorflow.keras as keras

from models.modules.modulated_conv import ModulatedConv

class MultichannelIamge(keras.Model):

    def __init__(
        self,
        in_ch,
        out_ch,
        style_dim,
        kernel_size = 1
    ):
        super(MultichannelIamge, self).__init__()
        self.conv = ModulatedConv(in_ch, out_ch, style_dim, kernel_size, demodulation=False)
        self.bias = tf.Variable(tf.zeros([1, out_ch, 1, 1]))

    def call(self, hidden, style):
        out = self.conv(hidden, style)
        out = out + self.bias
        return out
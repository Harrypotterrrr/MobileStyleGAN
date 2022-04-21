import tensorflow as tf
import tensorflow.keras as keras

from models.modules.utils import Upsample
from models.modules.modulated_conv import ModulatedConv


class ToRGB(keras.Model):

    def __init__(
            self,
            in_ch,
            style_dim,
            upsample = True,
            blur_kernel = [1, 3, 3, 1]
    ):
        super(ToRGB, self).__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv(in_ch, 3, 1, style_dim, demodulation=False)
        self.bias = tf.Variable(tf.zeros([1, 1, 1, 3]))

    def call(self, x, style, skip=None):
        out = self.conv(x, style)
        print("to_rgb")
        print(out.shape)
        out = out + self.bias
        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip
        return out

import tensorflow as tf

from tensorflow import nn
import tensorflow.keras as keras

from models.modules.utils import NoiseInjection
from models.modules.modulated_conv import ModulatedConv
from models.modules.fused_leakyReLU import FusedLeakyReLU

class StyledConv(keras.Model):

    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 style_dim,
                 upsample = False,
                 blur_kernel = [1, 3, 3, 1],
                 demodulation = True,
                 ):
        super(StyledConv, self).__init__()

        self.conv = ModulatedConv(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample = upsample,
            blur_kernel = blur_kernel,
            demodulation = demodulation,
        )
        self.noise_injection = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)
        # TODO: no bias?

    def call(self, x, style, noise = None):
        out = self.conv(x, style)
        out = self.noise_injection(out, noise=noise)
        out = self.activate(out)
        return out


class StyledConv2d(keras.Model):

    def __init__(
        self,
        channels_in,
        channels_out,
        style_dim,
        kernel_size,
        demodulate = True,
        conv_module = ModulatedConv
    ):
        super(StyledConv2d, self).__init__()

        self.conv = conv_module(
            channels_in,
            channels_out,
            style_dim,
            kernel_size,
            demodulate = demodulate
        )

        self.noise_injection = NoiseInjection()
        self.bias = tf.Variable(tf.zeros([1, channels_out, 1, 1]))
        self.act = nn.leaky_relu(0.2)

    def call(self, x, style, noise=None):
        out = self.conv(x, style)
        out = self.noise_injection(out, noise=noise)
        out = self.act(out + self.bias)
        return out
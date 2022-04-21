import math
import tensorflow as tf
import tensorflow.keras as keras

from models.modules.fused_leakyReLU import FusedLeakyReLU
from models.modules.utils import LinearTransform, LinearTransform2d, Blur

class ConvLayer(keras.Model):

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample = False,
        blur_kernel = [1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        self.layer_list = keras.Sequential()

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.layer_list.add(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        self.layer_list.add(
            LinearTransform2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                self.layer_list.add(FusedLeakyReLU(out_channel))

            else:
                pass
                # TODO ScaledLeakyRelu
                #layers.append(ScaledLeakyReLU(0.2))

    def call(self, x):
        out = self.layer_list(x)
        return out


class ResBlock(keras.Model):

    def __init__(self, in_ch, out_ch, blur_kernel=[1, 3, 3, 1]):
        super(ResBlock, self).__init__()

        self.conv1 = ConvLayer(in_ch, in_ch, 3)
        self.conv2 = ConvLayer(in_ch, out_ch, 3, downsample=True)
        self.skip = ConvLayer(
            in_ch, out_ch, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(out)

        skip = self.skip(x)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(keras.Model):

    def __init__(self,
                 size,
                 channels_in=3,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 activate=True
    ):

        super(Discriminator, self).__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.conv_seq = keras.Sequential(ConvLayer(channels_in, channels[size], 1))

        log_size = int(math.log(size, 2))
        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            self.conv_seq.add(ResBlock(in_channel, out_channel, blur_kernel))
            in_channel = out_channel

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = keras.Sequential(
            LinearTransform(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            LinearTransform(channels[4], 1),
        )
        self.activate = activate

    def forward(self, x):
        out = self.conv_seq(x) # TODO seq naming
        # out = self.minibatch_discrimination(out, self.stddev_group, self.stddev_feat) # TODO minibatch impl
        out = self.final_conv(out)
        out = tf.reshape(out, [out.shape[0], -1])
        out = self.final_linear(out)
        if self.activate:
            out = out.sigmoid()
        return {"out": out}



import math
import tensorflow as tf

from tensorflow import nn
from tensorflow import keras

from models.modules.upfirdn import upfirdn2d


def make_kernel(k):
    k = tf.Tensor(k, dtype=tf.float32)
    # if len(k.shape) == 1: # TODO get_shape
    if k.rank == 1:  # TODO get_shape
            k = k[None, :] * k[:, None]
    k /= k.sum()
    return k

class PixelNorm(keras.Model):

    def __init__(self):
        super(PixelNorm, self).__init__()

    def __call__(self, input):
        return input * tf.math.rsqrt(
            tf.reduce_mean(input ** 2, axis=1, keepdims=True)
            + 1e-8)

class EqualLinear(keras.Model):

    def __init__(
        self,
        in_dim,
        out_dim,
        bias=True,
        bias_init=0,
        lr_mul=1,
        activation=None,
        onnx_trace=False
    ):
        super(EqualLinear, self).__init__()

        self.weight = tf.Variable(tf.math.divide(
            tf.random.normal([out_dim, in_dim]),
            lr_mul
        ))

        if bias:
            self.bias = tf.Variable(tf.zeros(out_dim))
            # TODO fill with specified value
            #  self.bias = tf.Variable(tf.zeros(out_dim).fill(bias_init))
        else:
            self.bias = None
        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
        self.onnx_trace = onnx_trace

    def __call__(self, input):
        if self.activation:
            out = tf.matmul(input, self.weight * self.scale) # potential bug
            out = tf.nn.relu(out, self.bias * self.lr_mul, onnx_trace = self.onnx_trace)
            # TODO fused_leaky_relu
            #  out = fused_leaky_relu(out, self.bias * self.lr_mul, onnx_trace=self.onnx_trace)
        else:
            out = tf.matmul(input, self.weight * self.scale) + self.bias * self.lr_mul
        return out

    # TODO def repr inner function

class Blur(keras.Model):

    def __init__(self, kernel, pad, upsample_factor = 1):
        super(Blur, self).__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)
        self.pad = pad

    def __call__(self, input):
        # TODO CUDA upfirdn2d
        # out = upfirdn2d(input, self.kernel, pad=self.pad)
        out = upfirdn2d(input, self.kernel, pad=self.pad)
        return out

class ModulatedConv2d(keras.Model):

    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super(ModulatedConv2d, self).__init__()
        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = tf.Variable(
            tf.random.normal([1, out_channel, in_channel, kernel_size, kernel_size]
        ))
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __call__(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = tf.math.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.reshape(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.reshape(1, batch * in_channel, height, width)
            weight = weight.reshape(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = nn.conv2d_transpose(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.reshape(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.reshape(1, batch * in_channel, height, width)
            out = nn.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.reshape(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = nn.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out



class ConstantInput(keras.Model):

    def __init__(self, channel, size=4):
        super(ConstantInput, self).__init__()
        self.input = tf.Variable(tf.random.normal([1, channel, size, size]))

    def __call__(self, x):
        batch = x.shape[0]
        out = self.input.tile(batch, [1, 1, 1])
        return out

class Upsample(keras.Model):

    def __init__(self, kernel, factor=2):
        super(Upsample, self).__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel) # TODO register_buffer

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def __call__(self, input):
        return upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

class ToRGB(keras.Model):

    def __init__(
        self,
        in_channel,
        style_dim,
        upsample=True,
        blur_kernel=[1, 3, 3, 1]
    ):
        super(ToRGB, self).__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = tf.Variable(tf.zeros([1, 3, 1, 1]))

    def __call__(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias
        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip
        return out
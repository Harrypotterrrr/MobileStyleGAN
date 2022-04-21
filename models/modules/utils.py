import math
import tensorflow as tf
import tensorflow.keras as keras

from models.modules import *


def make_kernel(k):
    k = tf.constant(k, dtype=tf.float32)
    # if len(k.shape) == 1: # TODO get_shape
    if tf.rank(k) == 1:  # TODO get_shape
        k = k[None, :] * k[:, None]
    k /= tf.reduce_sum(k)
    return k


class PixelNorm(keras.Model):

    def __init__(self):
        super(PixelNorm, self).__init__()

    def call(self, x):
        return x * tf.math.rsqrt(
            tf.reduce_mean(x ** 2, axis=1, keepdims=True)
            + 1e-8
        )


class NonlinearTransform(keras.Model):

    def __init__(
            self,
            in_ch,
            out_ch,
            bias = True,
            bias_init = 0.0,
            lr_mul = 1,
            activation = None,
    ):
        super(NonlinearTransform, self).__init__()

        # TODO: lr_mul, math.sqrt(in_dim) meaning
        self.scale = (1 / math.sqrt(in_ch)) * lr_mul
        self.weight = tf.Variable(tf.math.divide(tf.random.normal([in_ch, out_ch]), lr_mul))
        self.weight = self.weight * self.scale

        if bias:
            self.bias = tf.Variable(tf.fill(in_ch, bias_init), dtype=tf.float32)
            self.bias = tf.reshape(self.bias, [1, out_ch])
            self.bias = self.bias * lr_mul
        else:
            self.bias = 0.0
        self.activation = activation

    def call(self, style):
        # style.shape (Batch, style_dim, in_channel)

        out = tf.matmul(style, self.weight) + self.bias
        # TODO fused_leaky_relu
        #  out = fused_leaky_relu(out, self.bias * self.lr_mul, onnx_trace=self.onnx_trace)
        if self.activation:
            out = tf.nn.leaky_relu(out, alpha=0.2)
        return out


class Blur(keras.Model):

    def __init__(self, kernel, pad, upsample_factor = 1):
        super(Blur, self).__init__()

        self.kernel = make_kernel(kernel)
        if upsample_factor > 1:
            self.kernel = self.kernel * (upsample_factor ** 2)
        self.pad = pad

    def call(self, x):
        # TODO CUDA upfirdn2d
        out = upfirdn2d(x, self.kernel, pad=self.pad)
        return out


class ConstantInput(keras.Model):

    def __init__(self, channel, size=4):
        super(ConstantInput, self).__init__()
        # TODO, change tf.Variable to tf.constant
        self.const_input = tf.constant(tf.random.normal([1, channel, size, size]))

    def call(self, x):
        out = tf.tile(self.const_input, [x.shape[0], 1, 1, 1])
        return out


class Upsample(keras.Model):

    def __init__(self, kernel, factor = 2):
        super(Upsample, self).__init__()

        self.factor = factor
        self.kernel = make_kernel(kernel) * (factor ** 2)

        pad = kernel.shape[0] - factor
        self.pad = ((pad + 1) // 2 + factor - 1, pad // 2)

    def call(self, x):
        return upfirdn2d(x, self.kernel, up=self.factor, down=1, pad=self.pad)


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

        self.conv = ModulatedConv(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = tf.Variable(tf.zeros([1, 3, 1, 1]))

    def call(self, x, style, skip=None):
        out = self.conv(x, style)
        out = out + self.bias
        if skip is not None:
            skip = self.upsample(skip)
            out = out + skip
        return out

class NoiseInjection(keras.Model):

    def __init__(self):
        super(NoiseInjection, self).__init__()
        # TODO weight zeros used for?
        self.weight = tf.Variable(tf.zeros(1))

    def call(self, x, noise=None):
        if noise is None:
            noise = tf.random.normal(x.shape)
        out = x + self.weight * noise
        return out
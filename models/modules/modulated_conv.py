import math
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow import nn

from models.modules.utils import Blur, LinearTransform


class ModulatedConv(keras.Model):

    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size,
            style_dim,
            demodulation = True,
            upsample = False,
            downsample = False,
            blur_kernel = [1, 3, 3, 1]
    ):
        super(ModulatedConv, self).__init__()

        self.eps = 1e-8
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blurring = Blur(blur_kernel, pad = (pad0, pad1), upsample_factor = factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2
            self.blurring = Blur(blur_kernel, pad = (pad0, pad1))


        self.padding = kernel_size // 2 # TODO: remove

        # Build weight for modulated conv
        fan_in = in_ch * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.weight = tf.Variable(
            tf.random.normal([1, out_ch, in_ch, kernel_size, kernel_size]
        ))
        self.weight = self.weight * self.scale

        self.modulation = LinearTransform(style_dim, in_ch, bias_init = 1.0)
        self.demodulation = demodulation

    def call(self, x, style):

        B, H, W, C_in = x.shape
        assert C_in == self.in_ch

        # linear transform of style to W in latent space
        style = self.modulation(style)
        style = tf.reshape(style, [B, 1, C_in, 1, 1]) # TODO: shape[1]?

        # scale weight by the magnitude self.scale
        w = self.weight * style # w.shape (B, out_ch, in_ch, K, K)
        print("self", self.out_ch, self.in_ch)
        print("self.weight", self.weight.shape)
        print('style.shape', style.shape)
        print("w", w.shape)

        if self.demodulation:
            demod_std = tf.math.rsqrt(tf.reduce_sum(w ** 2, axis = [2, 3, 4]) + self.eps)
            w = w * tf.reshape(demod_std, [B, self.out_ch, 1, 1, 1]) # weight.shape (B, out_ch, in_ch, K, K)
        if self.upsample:
            x = tf.transpose(x, [1, 2, 0, 3]) # x.shape (H, W, B*in_ch)
            x = tf.reshape(x, [1, H, W, B * C_in]) # x.shape (1, H, W, B*in_ch)
            w = tf.transpose(w, [3, 4, 1, 0, 2]) # weight.shape (K, K, out_ch, B, in_ch)
            w = tf.reshape(w, [self.kernel_size, self.kernel_size, self.out_ch, B * self.in_ch]) # weight.shape (K, K, out_ch, B*in_ch)
            print("?",x.shape)
            print("?",w.shape)
            print(B)
            print('in_ch', self.in_ch)
            print('out_ch', self.out_ch)

            out = nn.conv2d_transpose(x, w, output_shape=[1, H*2, W*2, self.out_ch], strides=2, padding='SAME') # out.shape (1, H, W, B*in_ch)
            print("out", out.shape)
            out = tf.transpose(out, [3, 1, 2, 0]) # out.shape (out_ch, H, W, 1)
            out = tf.reshape(out, [B, self.out_ch, out.shape[1], out.shape[2]]) # out.shape (B, out_ch, H, W)
            out = tf.transpose(out, [0, 2, 3, 1]) # out.shape (B, H, W, out_ch)
            # upsample the feature map
            out = self.blurring(out)

        elif self.downsample:
            # downsample the feature map
            x = self.blurring(x) # x.shape (B, H, W, C_in)

            B, H, W, _ = x.shape
            x = tf.transpose(x, [1, 2, 0, 3]) # x.shape (H, W, B, C_in)
            x = tf.reshape(x, [1, H, W, B * C_in]) # x.shape (1, H, W, B*C_in)
            w = tf.transpose(w, [3, 4, 0, 2, 1]) # weight.shape (K, K, B, in_ch, out_ch)
            w = tf.reshape(w, [self.kernel_size, self.kernel_size, self.in_ch, B * self.out_ch]) # weight.shape (K, K, in_ch, B*out_ch)

            out = nn.conv2d(x, w, strides=2, padding='SAME') # out.shape (1, H, W, B*out_ch)
            out = tf.transpose(out, [3, 1, 2, 0])  # out.shape (B*out_ch, H, W, 1)
            out = tf.reshape(out, [B, self.out_ch, out.shape[1], out.shape[2]]) # out.shape (B, out_ch, H, W)
            out = tf.transpose(out, [0, 2, 3, 1]) # out.shape (B, H, W, out_ch)

        else:
            x = tf.transpose(x, [1, 2, 0, 3])  # x.shape (H, W, B, C_in)
            x = tf.reshape(x, [1, H, W, B * C_in])  # x.shape (1, H, W, B*C_in)
            w = tf.transpose(w, [3, 4, 0, 2, 1])  # weight.shape (K, K, B, in_ch, out_ch)
            w = tf.reshape(w, [self.kernel_size, self.kernel_size, self.in_ch, B * self.out_ch])  # weight.shape (K, K, in_ch, B*out_ch)
            # TODO: use self.padding
            # out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            out = nn.conv2d(x, w, strides=1, padding='SAME')  # out.shape (1, H, W, B*out_ch)
            out = tf.transpose(out, [3, 1, 2, 0])  # out.shape (B*out_ch, H, W, 1)
            out = tf.reshape(out, [B, self.out_ch, out.shape[1], out.shape[2]])  # out.shape (B, out_ch, H, W)
            out = tf.transpose(out, [0, 2, 3, 1])  # out.shape (B, H, W, out_ch)

        return out
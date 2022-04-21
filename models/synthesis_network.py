import tensorflow as tf
import tensorflow.keras as keras

from models.modules.style_conv import StyledConv
from models.modules.utils import ConstantInput, ToRGB, Upsample


class SynthesisBlock(keras.Model):

    def __init__(self,
                 in_channel,
                 out_channel,
                 style_dim,
                 kernel_size = 3,
                 blur_kernel = [1, 3, 3, 1]
    ):
        super(SynthesisBlock, self).__init__()

        # Each synthesisBlock contains two styledConv
        self.conv1 = StyledConv(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample = True,
            blur_kernel = blur_kernel
        )
        self.conv2 = StyledConv(
            out_channel,
            out_channel,
            kernel_size,
            style_dim,
            blur_kernel = blur_kernel
        )

        self.to_rgb = ToRGB(out_channel, style_dim)

    def call(self, x, style, noise=[None, None]):
        if tf.rank(style) == 2:
            out = self.conv1(x, style, noise=noise[0])
            out = self.conv2(out, style, noise=noise[1])
            rgb = self.to_rgb(out, style)
        else:
            out = self.conv1(x, style[:, 0, :], noise=noise[0])
            out = self.conv2(out, style[:, 1, :], noise=noise[1])
            rgb = self.to_rgb(out,style[:, 2, :])
        return out, rgb


class SynthesisNetwork(keras.Model):

    def __init__(self,
        size,
        style_dim,
        blur_kernel = [1, 3, 3, 1],
        channels = [512, 512, 512, 512, 512, 256, 128, 64, 32] # optimize
    ):
        super().__init__()
        self.size = size
        self.style_dim = style_dim

        self.const_input = ConstantInput(channels[0])
        self.style_conv = StyledConv(
            channels[0], channels[0], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb = ToRGB(channels[0], style_dim, upsample=False)

        self.res_blocks = []
        in_channel = channels[0]

        # build residual block, for each is synthesisBlock with two styledConv
        for out_channel in channels[1:]:
            self.res_blocks.append(
                SynthesisBlock(
                    in_channel,
                    out_channel,
                    style_dim,
                    kernel_size = 3,
                    blur_kernel = blur_kernel
                )
            )
            in_channel = out_channel

        self.upsample = Upsample(blur_kernel)

    def call(self, style, noise_input = None):
        rtn = {"noise": [], "rgb": [], "img": None} # optimize

        # constant input of synthesis network
        const_input = self.const_input(style)

        # generate noise
        if noise_input is None:
            noise = tf.random.normal([1, 1, const_input.size(-1), const_input.size(-1)])
        else:
            noise = noise_input[0]
        rtn["noise"].append(noise)

        # const, style, noise fed into the first style block
        if tf.rank(style) == 2:
            hidden = self.style_conv(const_input, style, noise=noise)
            out = self.to_rgb(hidden, style)
        else:
            hidden = self.style_conv(const_input,style[:, 0, :], noise=noise)
            out = self.to_rgb(hidden,style[:, 1, :])
        rtn["rgb"].append(out)

        # residual blocks
        for i, res_block in enumerate(self.res_blocks):
            shape = [2, 1, 1, 2 ** (i + 3), 2 ** (i + 3)] # TODO: optimize
            if noise_input is None:
                noise = tf.random.normal(*shape)
            else:
                noise = noise_input[i + 1]
            hidden, rgb = res_block(hidden, style, noise)
            rtn["noise"].append(noise)
            rtn["rgb"].append(rgb)

            # short connection
            out = self.upsample(out) + rgb

        rtn["img"] = out
        return rtn

import tensorflow as tf
import tensorflow.keras as keras

from models.modules.style_conv import StyledConv, StyledConv2d
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
        self.conv1 = StyledConv(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=True,
            blur_kernel=blur_kernel
        )
        self.conv2 = StyledConv(
            out_channel,
            out_channel,
            kernel_size,
            style_dim,
            blur_kernel=blur_kernel
        )
        self.to_rgb = ToRGB(out_channel, style_dim)

    def call(self, x, style, noise=[None, None]):
        x = self.conv1(x, style if style.rank == 2 else style[:, 0, :], noise=noise[0])
        x = self.conv2(x, style if style.rank == 2 else style[:, 1, :], noise=noise[1])
        rgb = self.to_rgb(x, style if style.rank == 2 else style[:, 2, :])
        return x, rgb

class SynthesisNetwork(keras.Model):

    def __init__(self,
        size,
        style_dim,
        blur_kernel = [1, 3, 3, 1],
        channels = [512, 512, 512, 512, 512, 256, 128, 64, 32]
    ):
        super().__init__()
        self.size = size
        self.style_dim = style_dim

        self.const_input = ConstantInput(channels[0])
        self.conv1 = StyledConv(
            channels[0], channels[0], 3, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(channels[0], style_dim, upsample=False)

        self.models = []
        in_channel = channels[0]
        for out_channel in channels[1:]:
            self.models.append(
                SynthesisBlock(
                    in_channel,
                    out_channel,
                    style_dim,
                    3,
                    blur_kernel=blur_kernel
                )
            )
            in_channel = out_channel

        self.upsample = Upsample(blur_kernel)

    def call(self, style, noise=None):
        out = {"noise": [], "rgb": [], "img": None}

        const_input = self.const_input(style)
        if noise is None:
            # TODO to gpu
            # _noise = tf.random.normal([1, 1, hidden.size(-1), hidden.size(-1)]).to(style.device)
            _noise = tf.random.normal([1, 1, const_input.size(-1), const_input.size(-1)])
        else:
            _noise = noise[0]
        out["noise"].append(_noise)
        hidden = self.conv1(const_input, style if style.ndim == 2 else style[:, 0, :], noise=_noise)
        img = self.to_rgb1(hidden, style if style.ndim == 2 else style[:, 1, :])
        out["rgb"].append(img)

        for i, cur_layer in enumerate(self.models):
            shape = [2, 1, 1, 2 ** (i + 3), 2 ** (i + 3)]
            if noise is None:
                # TODO to CUDA
                # _noise = tf.random.normal(*shape).to(style.device)
                _noise = tf.random.normal(*shape)
            else:
                _noise = noise[i + 1]
            out["noise"].append(_noise)
            _style = style if style.ndim == 2 else style[:, 3*i+1:3*i+4, :]
            hidden, rgb = cur_layer(hidden, style, _noise)
            out["rgb"].append(rgb)
            img = self.upsample(img) + rgb

        out["img"] = img
        return out

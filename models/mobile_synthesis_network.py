import tensorflow as tf
import tensorflow.keras as keras

from models.modules.style_conv import StyledConv, StyledConv2d
from models.modules.utils import ConstantInput, ModulatedConv2d
from models.modules.multichannel_image import MultichannelIamge


class MobileSynthesisBlock(keras.Model):

    def __init__(
        self,
        channels_in,
        channels_out,
        style_dim,
        kernel_size=3,
        conv_module=ModulatedConv2d
    ):
        super(MobileSynthesisBlock, self).__init__()
        self.up = IDWTUpsaplme(channels_in, style_dim)
        self.conv1 = StyledConv2d(
            channels_in // 4,
            channels_out,
            style_dim,
            kernel_size,
            conv_module=conv_module
        )
        self.conv2 = StyledConv2d(
            channels_out,
            channels_out,
            style_dim,
            kernel_size,
            conv_module=conv_module
        )
        self.to_img = MultichannelIamge(
            channels_in=channels_out,
            channels_out=12,
            style_dim=style_dim,
            kernel_size=1
        )

    def call(self, x, style, noise=[None, None]):
        x = self.up(x, style if style.rank == 2 else style[:, 0, :])
        x = self.conv1(x, style if style.rank == 2 else style[:, 0, :], noise=noise[0])
        x = self.conv2(x, style if style.rank == 2 else style[:, 1, :], noise=noise[1])
        img = self.to_img(x, style if style.rank == 2 else style[:, 2, :])
        return x, img


class MobileSynthesisNetwork(keras.Model):
    def __init__(
        self,
        style_dim,
        channels = [512, 512, 512, 512, 512, 256, 128, 64]
    ):
        super(MobileSynthesisNetwork, self).__init__()
        self.style_dim = style_dim

        self.input_const = ConstantInput(channels[0])
        self.conv1 = StyledConv2d(
            channels[0],
            channels[0],
            style_dim,
            kernel_size=3
        )
        self.to_img1 = MultichannelIamge(
            channels_in=channels[0],
            channels_out=12,
            style_dim=style_dim,
            kernel_size=1
        )

        self.layers = []
        channels_in = channels[0]
        for i, channels_out in enumerate(channels[1:]):
            self.layers.append(
                MobileSynthesisBlock(
                    channels_in,
                    channels_out,
                    style_dim,
                    3,
                    conv_module=ModulatedDWConv2d
                )
            )
            channels_in = channels_out

        self.idwt = DWTInverse(mode="zero", wave="db1")
        self.register_buffer("device_info", torch.zeros(1))
        self.trace_model = False

    def call(self, style, noise=None):
        out = {"noise": [], "freq": [], "img": None}
        noise = NoiseManager(noise, self.device_info.device, self.trace_model)

        hidden = self.input_const(style)
        out["noise"].append(noise(hidden.size(-1)))
        hidden = self.conv1(hidden, style if style.ndim == 2 else style[:, 0, :], noise=out["noise"][-1])
        img = self.to_img1(hidden, style if style.ndim == 2 else style[:, 1, :])
        out["freq"].append(img)

        for i, m in enumerate(self.layers):
            out["noise"].append(noise(2 ** (i + 3), 2))
            _style = style if style.ndim == 2 else style[:, m.wsize()*i + 1: m.wsize()*i + m.wsize() + 1, :]
            hidden, freq = m(hidden, _style, noise=out["noise"][-1])
            out["freq"].append(freq)

        out["img"] = self.dwt_to_img(out["freq"][-1])
        return out

    def dwt_to_img(self, img):
        b, c, h, w = img.size()
        low = img[:, :3, :, :]
        high = img[:, 3:, :, :].view(b, 3, 3, h, w)
        return self.idwt((low, [high]))
import tensorflow as tf
import tensorflow.keras as keras

from models.modules.utils import PixelNorm, LinearTransform


class MappingNetwork(keras.Model):

    def __init__(
        self,
        style_dim,
        n_layers,
        lr_mlp = 0.01
    ):
        super(MappingNetwork, self).__init__()
        self.style_dim = style_dim
        # layer_list = [PixelNorm()]

        self.model = tf.keras.Sequential()
        self.model.add(PixelNorm())

        for i in range(n_layers):
            # layer_list.append(
            #     EqualLinear(style_dim, style_dim, lr_mul = lr_mlp, activation="fused_lrelu")
            # )
            self.model.add(LinearTransform(style_dim, style_dim, lr_mul = lr_mlp, activation="fused_lrelu"))

        # self.layer_list = keras.Sequential(layer_list)

    def call(self, x):
        return self.model(x)
        # return self.layer_list(x)

import tensorflow as tf
from tensorflow import keras

from models.modules.utils import PixelNorm, EqualLinear

class MappingNetwork(keras.Model):

    def __init__(
        self,
        style_dim,
        n_layers,
        lr_mlp = 0.01
    ):
        super(MappingNetwork, self).__init__()
        self.style_dim = style_dim
        layers = [PixelNorm()]
        for i in range(n_layers):
            layers.append(
                EqualLinear(style_dim, style_dim, lr_mul = lr_mlp, activation="fused_lrelu")
            )
        self.layers = keras.Sequential(*layers)

    def __call__(self, x):
        return self.layers(x)
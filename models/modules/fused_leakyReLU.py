import tensorflow as tf

from tensorflow import nn
import tensorflow.keras as keras


class FusedLeakyReLU(keras.Model):
    def __init__(
            self,
            channel,
            negative_slope=0.2,
            scale=2 ** 0.5,
    ):
        super().__init__()
        self.bias = tf.Variable(tf.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def call(self, input):
        # TODO: CUDA
        # return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale, self.trace_model)
        return nn.leaky_relu(input, 0.2)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5, trace_model=False):

    if len(tf.config.list_physical_devices('GPU')) != 0:
        # TODO GPU
        # return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)
        pass
        return None
    else: # cpu
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        # TODO: fixed ONNX conversion
        if trace_model:
            return nn.leaky_relu(
                input + bias.reshape(1, input.size(1)), negative_slope=0.2
            ) * scale
        else:
            return (
                    nn.leaky_relu(
                        input + bias.reshape(1, bias.shape[0], *rest_dim), negative_slope=0.2
                    ) * scale
            )
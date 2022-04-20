import tensorflow as tf
import tensorflow.keras as keras

from models.mapping_network import MappingNetwork
from models.synthesis_network import SynthesisNetwork

mapping_net_conf = {
    "style_dim": None,
    "n_layers": 0,
}


synthesis_net_conf = {
    "size": 1,
    "style_dim": 0,
}


class Distiller(keras.Model):

    def __init__(self):
        self.mapping_net = MappingNetwork(mapping_net_conf["style_dim"], mapping_net_conf["n_layers"])
        self.synthesis_net = SynthesisNetwork(synthesis_net_conf["size"], synthesis_net_conf["style_dim"])

    def call(self, x, truncated=False, generator="student"):

        # x to gpu
        #x = x.to(self.device_info.device)
        style = self.mapping_net(x)
        if truncated:
            style = self.style_mean + 0.5 * (style - self.style_mean)
        if generator == "student":
            img = self.student(style)["img"]
        else:
            img = self.synthesis_net(style)["img"]
        return img

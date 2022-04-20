import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import tensorflow.keras as keras

from utils.config import process_config
from models import *

def main():
    # config = process_config()
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(config["gpus"])

    # TODO: dataloader
    # dataloader = DataLoader(config)

    style_dim = 4
    B = 8
    n_layers = 16
    mapping_network = MappingNetwork(style_dim, n_layers)
    x = tf.random.normal([B, style_dim])
    out = mapping_network(x)
    print(out.shape)

    # TODO: trainer
    # trainer = Trainer(config, model, dataloader)
    # trainer.train()


if __name__ == '__main__':
    main()
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
import tensorflow.keras as keras

from utils.config import process_config
from models import *

def main():
    # config = process_config()
    # os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(config["gpus"])

    # TODO: dataloader
    # dataloader = DataLoader(config)

    style_dim = 8
    B = 2
    n_layers = 16
    style = tf.random.normal([B, style_dim])

    # Noise injection
    # noise = NoiseInjection()
    # x = tf.random.normal([4, 3, 32, 32])
    # y = noise(x)
    # print(y.shape)

    # make_kernel
    blur_kernel = [1, 3, 3, 1]
    factor = 2

    # x = tf.random.normal([B, 4, 4, 32])
    # style = tf.random.normal([B, style_dim])
    # mod = ModulatedConv(32, 64, 3, style_dim, upsample=True)
    # out = mod(x, style)
    # print(out.shape)

    # Linear Transform
    # us = Upsample(tf.constant([1, 3, 3, 1], dtype=tf.float32), 3)
    # x = tf.random.normal([2, 4, 4, 3])
    # out = us(x)
    # print(out.shape)




    # # sc = StyledConv(4, 8, 3, style_dim)
    # x = tf.random.normal([2, 32, 32, 4])
    # style = tf.random.normal([B, style_dim])
    # noise = tf.random.normal([B, 32])
    # #
    # sn = SynthesisNetwork(style_dim)
    # out = sn(style)
    # print(out.shape)

    # TODO: trainer
    # trainer = Trainer(config, model, dataloader)
    # trainer.train()


if __name__ == '__main__':
    main()
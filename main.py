import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from utils.config import process_config
from models import MappingNetwork

def main():
    config = process_config()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(config["gpus"])

    # TODO: dataloader
    # dataloader = DataLoader(config)

    model = MappingNetwork()

    # TODO: trainer
    # trainer = Trainer(config, model, dataloader)
    # trainer.train()


if __name__ == '__main__':
    main()
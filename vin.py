from model import Net
from config import VinConfig
import tensorflow as tf


def main():
    from  train import Trainer
    net=Net(VinConfig)
    net=net.cuda()
    net=net.double()
    trainer=Trainer(VinConfig,net)
    trainer.train()




if __name__ == '__main__':
    main()
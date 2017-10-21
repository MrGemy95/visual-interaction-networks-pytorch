from model import Net
from train import Trainer
from config import VinConfig


def main():
    net=Net(VinConfig).cuda().double()
    trainer=Trainer(VinConfig,net)
    trainer.train()




if __name__ == '__main__':
    main()
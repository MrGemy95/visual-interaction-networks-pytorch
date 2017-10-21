from vin_dataset import VinDataset, ToTensor
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from torch import nn
import numpy as np
import torch
from logger import Logger
from utils import create_dir
import os
class Trainer():
    def __init__(self, config, net):
        self.net = net
        self.config = config
        create_dir(self.config.checkpoint_dir)

        dataset = VinDataset(self.config, transform=ToTensor())
        self.dataloader = DataLoader(dataset, batch_size=self.config.batch_size,
                                     shuffle=True, num_workers=4)
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.0005)
        self.logger=Logger(self.config.log_dir)
        self.construct_cors()
        self.save()
        if config.load:
            self.load()


    def save(self):
        torch.save(self.net.state_dict(), os.path.join(self.config.checkpoint_dir,"checkpoint"))

    def load(self):
        self.net.load_state_dict(torch.load(os.path.join(self.config.checkpoint_dir,"checkpoint")))

    def construct_cors(self):
        # x-cor and y-cor setting
        nx, ny = (self.config.weight, self.config.height)
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        xv, yv = np.meshgrid(x, y)
        xv = np.reshape(xv, [self.config.height, self.config.weight, 1])
        yv = np.reshape(yv, [self.config.height, self.config.weight, 1])
        xcor = np.zeros((self.config.batch_size * 5, self.config.height, self.config.weight, 1), dtype=float)
        ycor = np.zeros((self.config.batch_size * 5, self.config.height, self.config.weight, 1), dtype=float)
        for i in range(self.config.batch_size * 5):
            xcor[i] = xv
            ycor[i] = yv
        xcor=xcor.transpose((0,3,1,2))
        ycor=ycor.transpose((0,3,1,2))
        self.xcor,self.ycor=Variable(torch.from_numpy(xcor)).cuda(), Variable(torch.from_numpy(ycor)).cuda()

    def train(self):
        total_step=0
        df = Variable(torch.ones(1)).double().cuda()
        for epoch in range(2):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(self.dataloader, 0):
                total_step+=1
                data=data[0]
                if data['image'].shape[0]<self.config.batch_size:
                    continue
                inputs, output_label, output_S_label = data['image'], data['output_label'], data[
                    'output_S_label']

                inputs, output_label, output_S_label = Variable(inputs).cuda(), Variable(output_label).cuda(), Variable(
                    output_S_label).cuda()

                self.optimizer.zero_grad()

                net_out, aux_out = self.net(inputs, self.xcor, self.ycor)

                # loss and optimizer
                loss = nn.MSELoss()
                mse = df * loss(net_out[0], output_label[:,0])

                for i in range(1, 8):
                    mse += (df ** (i + 1)) *loss(net_out[i], output_label[:,i])
                mse = mse / 8;
                ve_loss=loss(aux_out, output_S_label)
                total_loss = mse + ve_loss
                total_loss.backward()

                self.optimizer.step()


                #tensorboard_logging
                self.logger.scalar_summary("mse",mse.data[0],total_step,"train")
                self.logger.scalar_summary("ve_loss",ve_loss.data[0],total_step,"train")
                self.logger.scalar_summary("total_loss",total_loss.data[0],total_step,"train")
            print("epoch ",i," Finished")
        print('Finished Training')



    # def test(self):
    #     ts_img = np.zeros((1, int(frame_num), FLAGS.height, FLAGS.weight, FLAGS.col_dim), dtype=float);
    #     for i in range(1):
    #         for j in range(int(frame_num)):
    #             ts_img[i, j] = mpimg.imread(img_folder + "test/" + str(i) + "_" + str(j) + '.png')[:, :,
    #                            :FLAGS.col_dim];
    #     ts_data = np.zeros((1, int(frame_num), FLAGS.No * 5), dtype=float);
    #     for i in range(1):
    #         f = open(data_folder + "test/" + str(i) + ".csv", "r");
    #         ts_data[i] = [line[:-1].split(",") for line in f.readlines()];
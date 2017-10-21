from torch.utils.data import Dataset
import os
import numpy as np
import matplotlib.image as mpimg
import torch


class VinDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, config, transform=None):

        self.config = config
        self.transform = transform

        total_img = np.zeros((self.config.set_num, int(self.config.frame_num), self.config.height, self.config.weight,
                              self.config.col_dim))
        for i in range(self.config.set_num):
            for j in range(int(self.config.frame_num)):
                total_img[i, j] = mpimg.imread(self.config.img_folder + "train/" + str(i) + '_' + str(j) + '.png')[:, :,
                                  :self.config.col_dim]

        total_data = np.zeros((self.config.set_num, int(self.config.frame_num), self.config.No * 5))
        for i in range(self.config.set_num):
            f = open(self.config.data_folder + "train/" + str(i) + ".csv", "r")
            total_data[i] = [line[:-1].split(",") for line in f.readlines()]
        total_data = np.reshape(total_data, [self.config.set_num, int(self.config.frame_num), self.config.No, 5])

        # reshape img and data
        input_img = np.zeros((self.config.set_num * (int(self.config.frame_num) - 14 + 1), 6, self.config.height,
                              self.config.weight, self.config.col_dim)
                             )
        output_label = np.zeros((self.config.set_num * (int(self.config.frame_num) - 14 + 1), 8, self.config.No, 4)
                                )
        output_S_label = np.zeros((self.config.set_num * (int(self.config.frame_num) - 14 + 1), 4, self.config.No, 4)
                                  )
        for i in range(self.config.set_num):
            for j in range(int(self.config.frame_num) - 14 + 1):
                input_img[i * (int(self.config.frame_num) - 14 + 1) + j] = total_img[i, j:j + 6]
                output_label[i * (int(self.config.frame_num) - 14 + 1) + j] = np.reshape(total_data[i, j + 6:j + 14],
                                                                                         [8, self.config.No, 5])[
                                                                              :, :, 1:5]
                output_S_label[i * (int(self.config.frame_num) - 14 + 1) + j] = np.reshape(total_data[i, j + 2:j + 6],
                                                                                           [4, self.config.No, 5])[:, :,
                                                                                1:5]

        # shuffle
        tr_data_num = int(len(input_img) * 1)
        total_idx = np.arange(len(input_img))
        np.random.shuffle(total_idx)
        self.tr_data = input_img[total_idx]
        self.tr_label = output_label[total_idx]
        self.tr_S_label = output_S_label[total_idx]

    def __len__(self):
        return len(self.tr_data)

    def __getitem__(self, idx):

        sample = {'image': self.tr_data[idx], 'output_label': self.tr_label[idx],
                  'output_S_label': self.tr_S_label[idx]}

        if self.transform:
            sample = self.transform(sample),

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, output_label, output_S_label = sample['image'], sample['output_label'], sample[
            'output_S_label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((0, 3, 1, 2))
        return {'image': torch.from_numpy(image),
                'output_label': torch.from_numpy(output_label),
                'output_S_label': torch.from_numpy(output_S_label),
                }

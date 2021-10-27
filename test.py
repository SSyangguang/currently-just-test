"""Test file of enhancement and fusion"""
import os
import cv2
import glob
import torch
import random

import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from option import args
from dataloader import EnhanceTestData, EnhanceTestDataGray
from model import SeeInDark


class EnhanceTest(object):
    def __init__(self, batch_size=1):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.epochs = args.enhance_epochs
        self.low_data_names = args.test_lowimage_path
        self.high_data_names = args.test_highimage_path
        self.batch_num = len(self.low_data_names) // int(batch_size)
        self.patch = args.enhance_patch

        # load data and transform image to tensor and normalize
        self.test_set = EnhanceTestDataGray()
        self.test_loader = data.DataLoader(self.test_set, batch_size=batch_size,
                                           shuffle=True, num_workers=0, pin_memory=False)

        self.enhance_model = SeeInDark().to(self.device)
        self.state = torch.load(args.enhance_model_path + args.enhance_model)
        self.enhance_model.load_state_dict(self.state['model'])

    def test(self):

        for batch, (low_img, high_img) in enumerate(self.test_loader):

            low_img = low_img.to(self.device)
            high_img = high_img.to(self.device)

            output = self.enhance_model(low_img)

            low_img = low_img.cpu().detach().numpy()
            low_img = np.squeeze(low_img) * 255
            low_img = low_img.astype('uint8')
            # low_img = np.transpose(low_img, (1, 2, 0))

            output = output.cpu().detach().numpy()
            output = np.squeeze(output) * 255
            output = output.astype('uint8')
            # output = np.transpose(output, (1, 2, 0))

            high_img = high_img.cpu().detach().numpy()
            high_img = np.squeeze(high_img) * 255
            high_img = high_img.astype('uint8')
            # high_img = np.transpose(high_img, (1, 2, 0))

            plt.figure()
            plt.subplot(131)
            plt.imshow(low_img, cmap='gray')
            plt.axis('off')
            plt.subplot(132)
            plt.imshow(output, cmap='gray')
            plt.axis('off')
            plt.subplot(133)
            plt.imshow(high_img, cmap='gray')
            plt.axis('off')
            plt.show()


if __name__ == '__main__':
    model_test = EnhanceTest()
    model_test.test()

"""Training file of enhancement and fusion"""
import os
import glob
import torch
import random

import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from option import args
from dataloader import EnhanceTrainData, EnhanceTrainDataGray
from model import SeeInDark

seed = args.seed
random.seed(seed)
torch.manual_seed(seed)


class EnhanceTrain(object):
    def __init__(self):
        super(EnhanceTrain, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.epochs = args.enhance_epochs
        self.low_data_names = args.low_image_path
        self.high_data_names = args.high_image_path
        self.batch_size = args.enhance_batch
        self.batch_num = len(self.low_data_names) // int(self.batch_size)
        self.patch = args.enhance_patch

        # load data and transform image to tensor and normalize
        self.train_set = EnhanceTrainDataGray()
        self.train_loader = data.DataLoader(self.train_set, batch_size=self.batch_size,
                                            shuffle=True, num_workers=0, pin_memory=False)

        self.enhance_model = SeeInDark().to(self.device)

        self.lr = args.enhance_lr
        self.optimizer = optim.Adam(self.enhance_model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.l1_loss = torch.nn.L1Loss(reduction='mean').to(self.device)
        self.enhance_loss = []

    def train(self):
        writer = SummaryWriter(log_dir=args.enhance_log_dir, filename_suffix='enhance_train_loss')

        # load pre-trained enhancement model
        if os.path.exists(args.enhance_model):
            print('Loading pre-trained model')
            state = torch.load(args.enhance_model_path + args.enhance_model)
            self.enhance_model.load_state_dict(state)
            self.enhance_loss = state['enhance_train_loss']

        if not os.path.exists(args.enhance_model_path):
            os.makedirs(args.enhance_model_path)

        model = self.enhance_model
        model._initialize_weights()

        for epoch in range(0, self.epochs):
            l1_loss_epoch = []

            for batch, (low_img, high_img) in enumerate(self.train_loader):

                low_img = low_img.to(self.device)
                high_img = high_img.to(self.device)

                output = self.enhance_model(low_img)
                l1_loss = self.l1_loss(output, high_img)

                self.optimizer.zero_grad()
                l1_loss.backward()
                self.optimizer.step()

                l1_loss_epoch.append(l1_loss.item())

            # self.scheduler.step()
            self.enhance_loss.append(np.mean(l1_loss_epoch))
            print('epoch: %s, loss: %s' % (epoch, np.mean(l1_loss_epoch)))

            state = {
                'model': self.enhance_model.state_dict(),
                'enhance_train_loss': self.enhance_loss,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            torch.save(state, args.enhance_model_path + args.enhance_model)
            if epoch % 200 == 0:
                torch.save(state, args.enhance_model_path + str(epoch) + '.pth')

            writer.add_scalar('enhance_train_loss', np.mean(l1_loss_epoch), epoch)

        fig_mse, axe_mse = plt.subplots()
        axe_mse.plot(self.enhance_loss)
        fig_mse.savefig('train_loss_curve.png')
        print('Training finished')


if __name__ == '__main__':
    model_train = EnhanceTrain()
    model_train.train()

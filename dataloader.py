import glob
import torch
import random

import numpy as np
import torch.optim as optim
import torch.utils.data as data

from PIL import Image
from option import args


class EnhanceTrainData(data.Dataset):
    def __init__(self):
        super(EnhanceTrainData, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.low_data_path = args.low_image_path
        self.high_data_path = args.high_image_path
        self.batch_size = args.enhance_batch
        self.patch = args.enhance_patch

        # Get the image name list
        self.train_low_name = glob.glob(self.low_data_path + '/*.png')
        self.train_high_name = glob.glob(self.high_data_path + '/*.png')

    def __len__(self):
        assert len(self.train_low_name) == len(self.train_high_name)
        return len(self.train_low_name)

    def __getitem__(self, idx):
        # Get image and convert to HSV
        train_low_img = Image.open(self.train_low_name[idx])
        train_low_img = np.array(train_low_img, dtype='float32') / 255.0
        train_high_img = Image.open(self.train_high_name[idx])
        train_high_img = np.array(train_high_img, dtype='float32') / 255.0

        # crop and augmentation
        train_low_img, train_high_img = self.crops(train_low_img, train_high_img)
        train_low_img, train_high_img = self.augmentation(train_low_img, train_high_img)

        # Permute the images to tensor format
        train_low_img = np.transpose(train_low_img, (2, 0, 1))
        train_high_img = np.transpose(train_high_img, (2, 0, 1))

        # Return a patch
        self.input_low = train_low_img.copy()
        self.input_high = train_high_img.copy()

        return self.input_low, self.input_high

    def crops(self, train_low_img, train_high_img):
        # Take random crops
        h, w, _ = train_low_img.shape
        x = random.randint(0, h - self.patch)
        y = random.randint(0, w - self.patch)
        train_low_img = train_low_img[x: x + self.patch, y: y + self.patch, :]
        train_high_img = train_high_img[x: x + self.patch, y: y + self.patch, :]

        return train_low_img, train_high_img

    def augmentation(self, train_low_img, train_high_img):
        # Data augmentation
        if random.random() < 0.5:
            train_low_img = np.flipud(train_low_img)
            train_high_img = np.flipud(train_high_img)
        if random.random() < 0.5:
            train_low_img = np.fliplr(train_low_img)
            train_high_img = np.fliplr(train_high_img)
        rot_type = random.randint(1, 4)
        if random.random() < 0.5:
            train_low_img = np.rot90(train_low_img, rot_type)
            train_high_img = np.rot90(train_high_img, rot_type)

        return train_low_img, train_high_img


class EnhanceTestData(data.Dataset):
    def __init__(self):
        super(EnhanceTestData, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.low_data_path = args.test_lowimage_path
        self.high_data_path = args.test_highimage_path

        # Get the image name list
        self.test_low_name = glob.glob(self.low_data_path + '/*.png')
        self.test_high_name = glob.glob(self.high_data_path + '/*.png')

        self.patch = 256

    def __len__(self):
        assert len(self.test_low_name) == len(self.test_high_name)
        return len(self.test_low_name)

    def __getitem__(self, idx):
        # Get image
        test_low_img = Image.open(self.test_low_name[idx])
        test_low_img = np.array(test_low_img, dtype='float32') / 255.0
        test_high_img = Image.open(self.test_high_name[idx])
        test_high_img = np.array(test_high_img, dtype='float32') / 255.0

        # Permute the images to tensor format
        test_low_img = np.transpose(test_low_img, (2, 0, 1))
        test_high_img = np.transpose(test_high_img, (2, 0, 1))

        # Return image
        self.input_low = test_low_img.copy()
        self.input_high = test_high_img.copy()

        return self.input_low, self.input_high

    def crops(self, train_low_img, train_high_img):
        # Take random crops
        h, w, _ = train_low_img.shape
        x = random.randint(0, h - self.patch)
        y = random.randint(0, w - self.patch)
        train_low_img = train_low_img[x: x + self.patch, y: y + self.patch, :]
        train_high_img = train_high_img[x: x + self.patch, y: y + self.patch, :]

        return train_low_img, train_high_img


class EnhanceTrainDataGray(data.Dataset):
    def __init__(self):
        super(EnhanceTrainDataGray, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.low_data_path = args.low_image_path
        self.high_data_path = args.high_image_path
        self.batch_size = args.enhance_batch
        self.patch = args.enhance_patch

        # Get the image name list
        self.train_low_name = glob.glob(self.low_data_path + '/*.png')
        self.train_high_name = glob.glob(self.high_data_path + '/*.png')

    def __len__(self):
        assert len(self.train_low_name) == len(self.train_high_name)
        return len(self.train_low_name)

    def __getitem__(self, idx):
        # Get image and convert to HSV
        train_low_img = Image.open(self.train_low_name[idx])
        train_low_img = train_low_img.convert('HSV')
        train_low_img = np.array(train_low_img, dtype='float32') / 255.0
        train_high_img = Image.open(self.train_high_name[idx])
        train_high_img = train_high_img.convert('HSV')
        train_high_img = np.array(train_high_img, dtype='float32') / 255.0

        # Only get V space
        train_low_img = train_low_img[:, :, 2]
        train_low_img = train_low_img[:, :, np.newaxis]
        train_high_img = train_high_img[:, :, 2]
        train_high_img = train_high_img[:, :, np.newaxis]

        # crop and augmentation
        train_low_img, train_high_img = self.crops(train_low_img, train_high_img)
        train_low_img, train_high_img = self.augmentation(train_low_img, train_high_img)

        # Permute the images to tensor format
        train_low_img = np.transpose(train_low_img, (2, 0, 1))
        train_high_img = np.transpose(train_high_img, (2, 0, 1))

        # Return a patch
        self.input_low = train_low_img.copy()
        self.input_high = train_high_img.copy()

        return self.input_low, self.input_high

    def crops(self, train_low_img, train_high_img):
        # Take random crops
        h, w, _ = train_low_img.shape
        x = random.randint(0, h - self.patch)
        y = random.randint(0, w - self.patch)
        train_low_img = train_low_img[x: x + self.patch, y: y + self.patch]
        train_high_img = train_high_img[x: x + self.patch, y: y + self.patch]

        return train_low_img, train_high_img

    def augmentation(self, train_low_img, train_high_img):
        # Data augmentation
        if random.random() < 0.5:
            train_low_img = np.flipud(train_low_img)
            train_high_img = np.flipud(train_high_img)
        if random.random() < 0.5:
            train_low_img = np.fliplr(train_low_img)
            train_high_img = np.fliplr(train_high_img)
        rot_type = random.randint(1, 4)
        if random.random() < 0.5:
            train_low_img = np.rot90(train_low_img, rot_type)
            train_high_img = np.rot90(train_high_img, rot_type)

        return train_low_img, train_high_img


class EnhanceTestDataGray(data.Dataset):
    def __init__(self):
        super(EnhanceTestDataGray, self).__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.low_data_path = args.test_lowimage_path
        self.high_data_path = args.test_highimage_path

        # Get the image name list
        self.test_low_name = glob.glob(self.low_data_path + '/*.png')
        self.test_high_name = glob.glob(self.high_data_path + '/*.png')

        self.patch = 256

    def __len__(self):
        assert len(self.test_low_name) == len(self.test_high_name)
        return len(self.test_low_name)

    def __getitem__(self, idx):
        # Get image
        test_low_img = Image.open(self.test_low_name[idx])
        test_low_img = test_low_img.convert('HSV')
        test_low_img = np.array(test_low_img, dtype='float32') / 255.0
        test_high_img = Image.open(self.test_high_name[idx])
        test_high_img = test_high_img.convert('HSV')
        test_high_img = np.array(test_high_img, dtype='float32') / 255.0

        # Only get V space
        test_low_img = test_low_img[:, :, 2]
        test_low_img = test_low_img[:, :, np.newaxis]
        test_high_img = test_high_img[:, :, 2]
        test_high_img = test_high_img[:, :, np.newaxis]

        # Permute the images to tensor format
        test_low_img = np.transpose(test_low_img, (2, 0, 1))
        test_high_img = np.transpose(test_high_img, (2, 0, 1))

        # Return image
        self.input_low = test_low_img.copy()
        self.input_high = test_high_img.copy()

        return self.input_low, self.input_high

    def crops(self, train_low_img, train_high_img):
        # Take random crops
        h, w, _ = train_low_img.shape
        x = random.randint(0, h - self.patch)
        y = random.randint(0, w - self.patch)
        train_low_img = train_low_img[x: x + self.patch, y: y + self.patch, :]
        train_high_img = train_high_img[x: x + self.patch, y: y + self.patch, :]

        return train_low_img, train_high_img


if __name__ == '__main__':
    train_set = EnhanceTrainData()
    train_loader = data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=0, pin_memory=False)
    for batch, (low_img, high_img) in enumerate(train_loader):
        print(low_img)

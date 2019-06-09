import os
import torch
import cv2
import numpy as np
import PIL.Image as Image
import random
import scipy.io
from torch.utils import data
import torchvision.transforms as transforms


class VOCClassSeg(data.Dataset):
    class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])
    mean_rgb = [0.485, 0.456, 0.406]
    std_rgb = [0.229, 0.224, 0.225]
    stand_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_rgb, std=std_rgb)
    ])
    img_size = 500

    def __init__(self, root, split='train.txt', transform=False):
        self.root = root
        self.split = split

        dataset_dir = root + 'VOCdevkit/VOC2012/'
        imgsets_file = dataset_dir + 'ImageSets/Segmentation/' + split

        with open(imgsets_file) as f:
            imglist = f.readlines()

        data, label = [], []
        for img in imglist:
            img = img.strip()
            data.append(dataset_dir + 'JPEGImages/%s.jpg' % img)
            label.append(dataset_dir + 'SegmentationClass/%s.png' % img)
        self.data = data
        self.label = label
        self._transform = transform

    def __getitem__(self, idx):
        data_file = self.data[idx]
        label_file = self.label[idx]

        # load image
        img = cv2.imread(data_file)  # cv2 read bgr order, PIL.Image read rgb order
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # bgr -> rgb
        # assert img is not None


        label = Image.open(label_file)
        label = np.array(label, dtype=np.uint8)
        if self._transform:
            return self.transform(img, label)
        else:
            return img, label

    def transform(self, img, label):
        '''
        img: cv2 rgb
        '''
        img = self.stand_transform(img)
        label = torch.from_numpy(label).long()

        return img, label

    def untransform(self, img, label):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img = img * self.std_rgb + self.mean_rgb
        img *= 255
        img = img.astype(np.uint8)  # rgb
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        label = label.numpy()

        return img, label

    def randomFlip(self, img, label):
        if random.random() < 0.5:
            img = np.fliplr(img)
            label = np.fliplr(label)
        return img, label

    def resize(self, img, label, s):
        # print(s, img.shape)
        img = cv2.resize(img, (s, s), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (s, s), interpolation=cv2.INTER_NEAREST)
        return img, label

    def randomCrop(self, img, label):
        h, w, _ = img.shape
        short_size = min(w, h)
        rand_size = random.randrange(int(0.7 * short_size), short_size)
        x = random.randrange(0, w - rand_size)
        y = random.randrange(0, h - rand_size)

        return img[y:y + rand_size, x:x + rand_size], label[y:y + rand_size, x:x + rand_size]

    def __len__(self):
        return len(self.data)




if __name__ == '__main__':

    data = VOCClassSeg(root='../des/',
                       split='train.txt',
                       transform=True)

    x,y = data.__getitem__(4)



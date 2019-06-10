import torch
import torch.nn as nn
import os
import torch.utils.data as data
import cv2
import random
import numpy as np
from PIL import Image

class BaseDataset(data.Dataset):


    def __init__(
        self,
        root,
        split = 'train',
        ignore_label = 255,
        mean_bgr = [122.67,116.67,104.01],
        augment=True,
        base_size=None,
        crop_size=512,
        scales=[0.5, 0.75, 1.0, 1.25, 1.5],
        flip=True,
        year = 2012
    ):
        self.root = root
        self.split = split
        self.ignore_label = ignore_label
        self.mean_bgr = np.array(mean_bgr)
        self.augment = augment
        self.base_size = base_size
        self.crop_size = crop_size
        self.scales = scales
        self.flip = flip
        self.files = []
        self.year = year
        self._set_files()

        cv2.setNumThreads(0)

    def _set_files(self):

        raise NotImplementedError()

    def _load_data(self, image_id):

        raise NotImplementedError()

    def _augmentation(self, image, label):
        # Scaling
        h, w = label.shape
        if self.base_size:
            if h > w:
                h, w = (self.base_size, int(self.base_size * w / h))
            else:
                h, w = (int(self.base_size * h / w), self.base_size)
        scale_factor = random.choice(self.scales)
        h, w = (int(h * scale_factor), int(w * scale_factor))
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)
        label = np.asarray(label, dtype=np.int64)

        # Padding to fit for crop_size
        h, w = label.shape
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT,
        }
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, value=self.mean_bgr, **pad_kwargs)
            label = cv2.copyMakeBorder(label, value=self.ignore_label, **pad_kwargs)

        # Cropping
        h, w = label.shape
        start_h = random.randint(0, h - self.crop_size)
        start_w = random.randint(0, w - self.crop_size)
        end_h = start_h + self.crop_size
        end_w = start_w + self.crop_size
        image = image[start_h:end_h, start_w:end_w]
        label = label[start_h:end_h, start_w:end_w]

        if self.flip:
            # Random flipping
            if random.random() < 0.5:
                image = np.fliplr(image).copy()  # HWC
                label = np.fliplr(label).copy()  # HW
        return image, label

    def __getitem__(self, index):
        image_id, image, label = self._load_data(index)
        if self.augment:
            image, label = self._augmentation(image, label)
        # Mean subtraction
        image -= self.mean_bgr
        # HWC -> CHW
        image = image.transpose(2, 0, 1)
        return image.astype(np.float32), label.astype(np.int64)

    def __len__(self):
        return len(self.files)



class VOC_load(BaseDataset):

    def __init__(self,**kwargs):
        super(VOC_load,self).__init__(**kwargs)
    def _set_files(self):
        self.root = os.path.join(self.root,'VOC{}'.format(self.year))
        self.img_dir = os.path.join(self.root, "JPEGImages")
        self.gt_dir = os.path.join(self.root, "SegmentationClass")
        if self.split in ["train", "trainval", "val", "test"]:
            files = os.path.join(self.root,"ImageSets/Segmentation",self.split + '.txt')
            files = tuple(open(files,'r'))
            files = [i.rstrip() for i in files]
            self.files = files
        print(self.files)

    def _load_data(self, image_id):
        id = self.files[image_id]
        im_path = os.path.join(self.img_dir,id + '.jpg')
        print(im_path)
        label_path = os.path.join(self.gt_dir,id + '.png')
        im = cv2.imread(im_path,cv2.IMREAD_COLOR).astype(np.float32)
        label = np.asarray(Image.open(label_path),dtype=np.int32)
        return id,im,label



if __name__ == '__main__':
    test_data = VOC_load(year=2012,root='../des/VOCdevkit')
    x,y = test_data.__getitem__(5)
    y = y.reshape(512,512,1)
    label = y.astype(np.uint8)

    # import matplotlib.pyplot as plt
    # from torchvision.utils import make_grid

    # print(x.shape,y)
    # x = np.transpose(x,(1,2,0))
    cv2.imshow('2',label)
    cv2.waitKey(0)
    # loader = data.DataLoader(test_data, batch_size=1)
    # for i,(images, labels) in enumerate(loader):
    #     if i == 0:
    #         labels = labels[:, np.newaxis, ...]
    #         label = make_grid(labels, pad_value=255).numpy()
    #         label_ = np.transpose(label, (1, 2, 0))[..., 0].astype(np.float32)
    #         label = cm.jet_r(label_ / 21.0) * 255
    #         mask = np.zeros(label.shape[:2])
    #         label[..., 3][(label_ == 255)] = 0
    #         label = label.astype(np.uint8)
    #         cv2.imshow('1',label)
    #         cv2.waitKey(0)




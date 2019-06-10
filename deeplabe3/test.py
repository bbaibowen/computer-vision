from network import get_deeplab3
import torch
import numpy as np
import cv2
from utils import labelTopng



PATH = 'C:\\Users\\ZD\\Desktop\\deeplabv3plus-pytorch-master\\deeplabv3plus_xception_VOC2012_epoch46_all.pth'
def read_im(path):

    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) * (1./250.)
    im = cv2.resize(im,(512,512))

    return im

model = get_deeplab3(PATH,train = False)
imgs = []
imgs += [read_im('../1.jpg')]
imgs += [read_im('../3.jpg')]
imgs = np.array(imgs,dtype=float)
imgs = np.transpose(imgs,(0,3,1,2))
imgs = torch.Tensor(imgs)
outs = model(imgs)
for i,out in enumerate(outs):
    label = out.unsqueeze(0).data.max(1)[1].squeeze_(1).squeeze_(0)
    labelTopng(label, img_name='test{}.png'.format(i))


import torch
import torch.nn as nn
import cv2
import numpy as np
from net import FCN8s
from utils import trans_im,labelTopng


im = cv2.imread('../person.jpg')
im = np.transpose(im,(2,0,1))
im = np.expand_dims(im,0)
im = torch.Tensor(im)
PATH = 'C:\\Users\\ZD\\Desktop\\fcn8s_from_caffe.pth'
load = torch.load(PATH, 'cpu')
model = FCN8s()
model.load_state_dict(load)
pred = model(im)
label = pred.data.max(1)[1].squeeze_(1).squeeze_(0)
labelTopng(label,img_name = 'test.png')





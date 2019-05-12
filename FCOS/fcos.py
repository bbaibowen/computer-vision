import torch.nn as nn
import torch
import numpy as np
import cv2
from resnet import model
from net_pred import end_pred
from collections import OrderedDict



class FCOS(nn.Module):

    def __init__(self):

        super(FCOS,self).__init__()


        self.backbone = model
        self.rpn = end_pred(self.backbone.out_channels)
        print(self.backbone.out_channels)

    def forward(self,x):

        feats = self.backbone(x)
        cls, box, centerness,locs = self.rpn(feats)
        return cls,box,centerness,locs

fcos = FCOS()
models = nn.Sequential(OrderedDict([("module",fcos)]))



if __name__ == '__main__':
    imgs = np.array(cv2.resize(cv2.imread('../3.jpg'), (1024, 800)))
    imgs = np.expand_dims(imgs, 0)
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    x = torch.FloatTensor(imgs)

    cls, box, centerness,locs = models(x)
    for i in models.state_dict():
        print(i)
    for i,j in enumerate(cls):
        print('cls',cls[i].shape)
        print('box',box[i].shape)
        print('center',centerness[i].shape)
        print('loc',locs[i].shape)





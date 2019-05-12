import torch
import torch.nn as nn
import numpy as np
import math
#FPN
import torch
import torch.nn.functional as F
from torch import nn



CLASS = 81


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class Head(nn.Module):
    def __init__(self,in_channel):
        super(Head,self).__init__()
        cls = list()
        loc = list()

        for i in range(4):
            cls.append(nn.Conv2d(in_channel,in_channel,kernel_size=3,stride=1,padding=1))
            cls.append(nn.GroupNorm(32,in_channel))
            cls.append(nn.ReLU())
            loc.append(nn.Conv2d(in_channel,in_channel,kernel_size=3,stride=1,padding=1))
            loc.append(nn.GroupNorm(32,in_channel))
            loc.append(nn.ReLU())
        self.add_module('cls_tower',nn.Sequential(*cls))
        self.add_module('bbox_tower',nn.Sequential(*loc))
        self.cls_logits = nn.Conv2d(in_channel,CLASS-1,kernel_size=3,stride=1,padding=1)
        self.bbox_pred = nn.Conv2d(in_channel,4,kernel_size=3,stride=1,padding=1)
        self.centerness = nn.Conv2d(in_channel,1,kernel_size=3,stride=1,padding=1)
        self.scales = nn.ModuleList([Scale(init_value=1.) for i in range(5)])
        self.initialization()


    def initialization(self):
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    nn.init.normal_(l.weight, std=0.01)
                    nn.init.constant_(l.bias, 0)
        bias_ = -math.log((1 - 0.01) / 0.01)
        nn.init.constant_(self.cls_logits.bias,bias_)

    def forward(self, x):
        logits = list()
        bbox = list()
        centerness = list()

        for i,j in enumerate(x):

            cls = self.cls_tower(j)
            logits.append(self.cls_logits(cls))
            centerness.append(self.centerness(cls))
            bbox.append(torch.exp(self.scales[i](self.bbox_pred(self.bbox_tower(j)))))

        return logits,bbox,centerness

class end_pred(nn.Module):

    def __init__(self,in_channels):
        super(end_pred,self).__init__()

        self.head = Head(in_channels)
        self.stride = [8, 16, 32, 64, 128]


    def forward(self, x,is_train = False):

        cls,box,centerness = self.head(x)

        #这个东西是feature map所有像素，然后映射回原图，嗯就是这样
        loctions = list()
        for i,j in enumerate(x):
            h,w = j.size()[-2:]
            loc_level = self.compute_locations_per_level(h,w,self.stride[i],j.device)
            loctions.append(loc_level)

        if is_train:
            pass
        else:


            return cls,box,centerness,loctions

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


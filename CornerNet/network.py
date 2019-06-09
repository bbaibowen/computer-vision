import torch
import torch.nn as nn
import numpy as np
import cv2
from corner_pooling import *

def get_tl_pool(channels):

    return pool(channels=channels,pool1=top_pooling,pool2=left_pooling)

def get_br_pool(channels):

    return pool(channels=channels,pool1=bottom_pooling,pool2=right_pooling)


class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu


class residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.skip = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        skip = self.skip(x)
        return self.relu(bn2 + skip)


def hourglass_layer(k_size,in_channels,out_channels,iter):

    layers = [residual(k_size,in_channels,out_channels,stride=2)]
    layers += [residual(k_size,out_channels,out_channels) for i in range(iter - 1)]

    return nn.Sequential(*layers)

def make_layer(k, inp_dim, out_dim, modules, layer):
    layers = [layer(k, inp_dim, out_dim)]
    for _ in range(1, modules):
        layers.append(layer(k, out_dim, out_dim))
    return nn.Sequential(*layers)

def make_layer_revr(k, inp_dim, out_dim, modules, layer):
    layers = []
    for _ in range(modules - 1):
        layers.append(layer(k, inp_dim, inp_dim))
    layers.append(layer(k, inp_dim, out_dim))
    return nn.Sequential(*layers)

n = 5
in_channels = [256, 256, 384, 384, 384, 512]
modules = [2, 2, 2, 2, 2, 4]
out_channels = 80  #class num
nstack = 2
cnv_dim = 256

class hourglass_backbone(nn.Module):

    def __init__(self,k,dim,module):
        super(hourglass_backbone,self).__init__()
        self.n = k
        curr_mod = module[0]
        next_mod = module[1]

        curr_dim = dim[0]
        next_dim = dim[1]

        self.up1 = make_layer(3,curr_dim,curr_dim,curr_mod,residual)
        self.low1 = hourglass_layer(3,curr_dim,next_dim,curr_mod)
        if self.n > 1:
            self.low2 = hourglass_backbone(k = k - 1,dim = dim[1:],module = module[1:])
        else:
            self.low2 = make_layer(3,next_dim,next_dim,next_mod,residual)
        self.low3 = make_layer_revr(3,next_dim,curr_dim,curr_mod,residual)
        self.up2 = nn.Upsample(scale_factor=2)

    def forward(self, x):

        up1 = self.up1(x)
        low1 = self.low1(x)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)

        return up1 + up2


hourglass = hourglass_backbone(k=n,dim=in_channels,module=modules)


class Model(nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        #res
        self.pre = nn.Sequential(convolution(7,3,128,stride=2),residual(3,128,256,stride=2))
        self.kps = nn.ModuleList([hourglass for i in range(nstack)])
        self.cnvs = nn.ModuleList([convolution(3,in_channels[0],cnv_dim) for i in range(nstack)])

        #cornerpool pool
        self.tl_cnvs = nn.ModuleList([get_tl_pool(cnv_dim) for i in range(nstack)])
        self.br_cnvs = nn.ModuleList([get_br_pool(cnv_dim) for i in range(nstack)])

        #heatmap
        self.tl_heats = nn.ModuleList([
            nn.Sequential(
                convolution(3, cnv_dim, in_channels[0], with_bn=False),
                nn.Conv2d(in_channels[0], 80, (1, 1))) for i in range(nstack)
        ])
        self.br_heats = nn.ModuleList([
            nn.Sequential(
                convolution(3, cnv_dim, in_channels[0], with_bn=False),
                nn.Conv2d(in_channels[0], 80, (1, 1))) for i in range(nstack)
        ])

        #embedding
        self.tl_tags = nn.ModuleList([
            nn.Sequential(
                convolution(3, cnv_dim, in_channels[0], with_bn=False),
                nn.Conv2d(in_channels[0], 1, (1, 1))) for i in range(nstack)
        ])
        self.br_tags = nn.ModuleList([
            nn.Sequential(
                convolution(3, cnv_dim, in_channels[0], with_bn=False),
                nn.Conv2d(in_channels[0], 1, (1, 1))) for i in range(nstack)
        ])

        self.inters = nn.ModuleList([
            residual(3,in_channels[0],in_channels[0]) for i in range(nstack - 1)
        ])

        self.inters_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels[0], in_channels[0], (1, 1), bias=False),
                nn.BatchNorm2d(in_channels[0])
            ) for _ in range(nstack - 1)
        ])

        self.cnvs_ = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(cnv_dim, in_channels[0], (1, 1), bias=False),
                nn.BatchNorm2d(in_channels[0])
            ) for _ in range(nstack - 1)
        ])

        #offset
        self.tl_regrs = nn.ModuleList([
            nn.Sequential(
                convolution(3, cnv_dim, in_channels[0], with_bn=False),
                nn.Conv2d(in_channels[0], 2, (1, 1))
            ) for i in range(nstack)
        ])

        self.br_regrs = nn.ModuleList([
            nn.Sequential(
                convolution(3, cnv_dim, in_channels[0], with_bn=False),
                nn.Conv2d(in_channels[0], 2, (1, 1))
            ) for i in range(nstack)
        ])

    def forward(self, x):

        inter = self.pre(x)
        outs = []
        layers = zip(
            self.kps,self.cnvs,
            self.tl_cnvs,self.br_cnvs,
            self.tl_heats,self.br_heats,
            self.tl_tags,self.br_tags,
            self.tl_regrs,self.br_regrs
        )

        for ind ,layer in enumerate(layers):


            kp_,cnv_ = layer[0:2]
            tl_cnv_, br_cnv_ = layer[2:4]
            tl_heat_, br_heat_ = layer[4:6]
            tl_tag_, br_tag_ = layer[6:8]
            tl_regr_, br_regr_ = layer[8:10]

            kp = kp_(inter)
            cnv = cnv_(kp)
            if ind == nstack - 1:
                tl_cnv = tl_cnv_(cnv)
                br_cnv = br_cnv_(cnv)
                tl_heat, br_heat = tl_heat_(tl_cnv), br_heat_(br_cnv)
                tl_tag, br_tag = tl_tag_(tl_cnv), br_tag_(br_cnv)
                tl_regr, br_regr = tl_regr_(tl_cnv), br_regr_(br_cnv)

                outs += [tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr]

            if ind < nstack - 1:
                inter = self.inters_[ind](inter) + self.cnvs_[ind](cnv)
                inter = nn.ReLU(inplace=True)(inter)
                inter = self.inters[ind](inter)


        return outs

if __name__ == '__main__':

    a = torch.load('CornerNet_500000.pkl','cpu')
    for i,j in enumerate(a):
        print(j)
    img = cv2.imread('../3.jpg')
    img = cv2.resize(img, (511, 511))
    img = np.expand_dims(img, 0)
    img = np.transpose(img, (0, 3, 2, 1))
    img = torch.Tensor(img)
    print(img.shape)
    model = Model()
    out = model(img)
    print('-------------------------------------------------')
    for i in model.state_dict():
        print(i)

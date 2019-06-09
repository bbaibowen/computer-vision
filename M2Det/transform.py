import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2

#这里的话只搭建网络，剩下的部分和我之前写过的SSD/RFB等没有什么区别
#backbone：vgg,为了更直观一点，我采用TF/keras的写法



IMG_SIZE = 512

class CBR(nn.Module):

    def __init__(self,in_channel,out_channel,k,s = 1,padding = 0,dilation = 1,groups=1,is_relu = True,is_bn = True,bias = False):
        super(CBR,self).__init__()
        self.is_bn = is_bn
        self.is_relu = is_relu
        self.conv2d = nn.Conv2d(in_channel,out_channel,kernel_size=k,stride=s,padding=padding,dilation=dilation,groups=groups,bias=bias)
        self.bn = nn.BatchNorm2d(out_channel,eps=1e-5,momentum=1e-2,affine=True)
        self.prelu = nn.PReLU()

    def forward(self, x):
        net = self.conv2d(x)
        net = self.bn(net) if self.is_bn else net
        net = self.prelu(net) if self.is_relu else net

        return net



def build_vgg(x):

    #b1  3
    x = CBR(3,64,3,padding=1,is_bn=False)(x)
    x = CBR(64,64,3,padding=1,is_bn=False)(x)
    x = nn.MaxPool2d(kernel_size=2,stride=2)(x)

    #b2 3
    x = CBR(64, 128, 3, padding=1, is_bn=False)(x)
    x = CBR(128, 128, 3, padding=1, is_bn=False)(x)
    x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

    #b3 4
    x = CBR(128, 256, 3, padding=1, is_bn=False)(x)
    x = CBR(256, 256, 3, padding=1, is_bn=False)(x)
    x = CBR(256, 256, 3, padding=1, is_bn=False)(x)
    x = nn.MaxPool2d(kernel_size=2, stride=2,ceil_mode=True)(x)

    #b4 4
    x = CBR(256, 512, 3, padding=1, is_bn=False)(x)
    x = CBR(512, 512, 3, padding=1, is_bn=False)(x)
    x = CBR(512, 512, 3, padding=1, is_bn=False)(x)
    out1 = x
    x = nn.MaxPool2d(kernel_size=2, stride=2)(x)

    #b5 3
    x = CBR(512, 512, 3, padding=1, is_bn=False)(x)
    x = CBR(512, 512, 3, padding=1, is_bn=False)(x)
    x = CBR(512, 512, 3, padding=1, is_bn=False)(x)


    x = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)(x)
    x = CBR(512,1024,3,padding=6,dilation=6,is_bn=False)(x)
    out2 = CBR(1024,1024,1,is_bn=False)(x)

    return out1,out2

def FFM1(x1,x2):

    x1 = CBR(512,256,3,padding=1)(x1)
    x2 = F.interpolate(CBR(1024,512,1)(x2),scale_factor=2)
    return torch.cat([x1,x2],dim=1)

def upsampling(x1, x2):
    _, _, h, w = x2.size()

    return F.interpolate(x1, size=(h, w)) + x2

def FFM2(x1,x2):

    x1 = CBR(512 + 256,128,1,is_bn=False,is_relu=False)(x1)
    return torch.cat([x1,x2],dim=1)

def TUM(x,idx):

    # conv = [CBR(x.size(1),128)]
    conv = [x] if idx != 0 else [CBR(x.size(1),256,1)(x)]
    # print(x.shape)
    layers = []

    #conv
    if idx == 0:
        x = CBR(x.size(1),256,3,2,1)(x)
        conv.append(x)
        for i in range(4):
            x = CBR(256,256,3,2,1)(x)
            conv.append(x)

    else:
        for i in range(5):
            x = CBR(256, 256, 3, 2, 1)(x)
            conv.append(x)


    #deconv and transform
    for i,j in enumerate(conv[::-1]):

        if i == 0:

            layers += [CBR(j.size(1),128,1)(j)]
            x = CBR(j.size(1),256,3,padding=1)(j)
        else:
            x = upsampling(x,j)
            layers.append(CBR(x.size(1),128,1)(x))
            x = CBR(x.size(1),256,3,padding=0)(x)


    return layers


def SFAM(x):

    x = np.array(x)
    merge = list()
    for i in range(6):

        net = torch.cat(list(x[:, i]), dim=1)
        mul = net
        net = nn.AdaptiveAvgPool2d(1)(net)
        net = nn.Conv2d(128 * 8,128 * 8 // 16,1)(net)
        net = nn.ReLU(inplace=True)(net)
        net = nn.Conv2d(128 * 8 // 16,128 * 8,1)(net)
        merge += [nn.Sigmoid()(net) * mul]

    return merge



def build_network(x):

     vgg_out = build_vgg(x)#[batch,512,64,64],[batch,1024,32,32]
     all_fpn = []
     #FFM1 merge
     merge = FFM1(vgg_out[0],vgg_out[1]) #channel 512+256

     ######################    TUM     ####################
     #源码中用的8个TUM，且第一个没有FFM2
     for i in range(8):
         if i == 0:
             layers = TUM(merge,0)
             all_fpn.append(layers)
         else:
             layers = TUM(FFM2(merge,layers[-1],),i)
             all_fpn.append(layers)


     #########################  SFAM  ##########################
     sfam = SFAM(all_fpn)

if __name__ == '__main__':
    imgs = np.array(cv2.resize(cv2.imread('../3.jpg'), (512, 512)))
    imgs = np.expand_dims(imgs, 0)
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    x = torch.FloatTensor(imgs)
    build_network(x)
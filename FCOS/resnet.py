import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
from fpn import FPN,LastLevelMaxPool,LastLevelP6P7
from collections import OrderedDict

# FPN AND ResNet
#所有的initialization、nn.init.kaiming_normal都是参数初始化

# dic = torch.load('FCOS_R_50_FPN_1x.pth','cpu')
# for i,j in enumerate(dic['model']):
#     print(j)



def conv3x3(in_channel,out_channel,stride = 1):

    return nn.Conv2d(in_channel,out_channel,kernel_size=3,stride=stride,padding=1,bias=False)

class BasicBlock(nn.Module):

    def __init__(self,in_channel,out_channel,stride = 1,downsample = None):
        super(BasicBlock,self).__init__()
        self.conv1 = conv3x3(in_channel,out_channel,stride)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(in_channel,out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.dowsample = downsample
        self.stride = stride

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.dowsample is not None:
            residual = self.dowsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,dilated = 1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel,out_channel,kernel_size=3,stride=stride,padding=1,bias=False,dilation=dilated)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(out_channel,out_channel * 4,kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Stem(nn.Module):

    def __init__(self,out_channel):
        super(Stem,self).__init__()
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        for i in [self.conv1,]:
            nn.init.kaiming_uniform_(i.weight,a=1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        return x

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64
        #STEM
        self.stem = Stem(64)

        #layer
        self.layer1 = self._make_layer(block,64,layers[0])
        self.layer2 = self._make_layer(block,128,layers[1],stride=2)
        self.layer3 = self._make_layer(block,256,layers[2],stride=2)
        self.layer4 = self._make_layer(block,512,layers[3],stride=2)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal(m.weight,mode = 'fan_out',nonlinearity = 'relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * 4,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * 4
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,))
        return nn.Sequential(*layers)


    def forward(self, x):


        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        s2 = x
        x = self.layer3(x)
        s3 = x
        x = self.layer4(x)


        return s2,s3,x

def resnet50():

    model = ResNet(Bottleneck,[3,4,6,3])
    # torch.save(model.state_dict(),'./res.pth')
    return model

def build():

    def head():
        def conv_uniform(in_channels, out_channels, k_size, stride=1, dilated=1, is_bn=False):

            conv = nn.Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride,
                             padding=dilated * (k_size - 1) // 2, dilation=dilated, bias=False if is_bn else True)
            nn.init.kaiming_uniform_(conv.weight, a=1)
            module = [conv, ]
            if is_bn:
                module.append(nn.BatchNorm2d(out_channels))
            if len(module) > 1:
                return nn.Sequential(*module)
            return conv
        return conv_uniform

    body = resnet50()
    in_channels_ = 256
    out_channels = 256 * 4

    in_channel_p6p7 = in_channels_ * 8
    fpn = FPN(in_channels_list=[
            0,
            in_channels_ * 2,
            in_channels_ * 4,
            in_channels_ * 8,
        ],out_channels = out_channels,conv_block = head(),
        top_blocks=LastLevelP6P7(in_channel_p6p7, out_channels))
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


model = build()



if __name__ == '__main__':
    imgs = np.array(cv2.resize(cv2.imread('../3.jpg'), (512, 512)))
    imgs = np.expand_dims(imgs, 0)
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    x = torch.FloatTensor(imgs)

    # model = build()
    # for i in model.state_dict():
    #     print(i)
    #
    # feats = model(x)
    # for i in feats:
    #     print(i.shape)

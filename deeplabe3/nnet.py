import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class ASPP(nn.Module):


    def __init__(self,in_channels,out_channels,rate = None):

        self.inter = out_channels * 5

        super(ASPP,self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1,1,padding=0,dilation=rate[0]),
            nn.BatchNorm2d(out_channels),
            # SynchronizedBatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=rate[1], dilation=rate[1]),
            # SynchronizedBatchNorm2d(out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding = rate[2], dilation=rate[2]),
            # SynchronizedBatchNorm2d(out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=rate[-1], dilation=rate[-1]),
            # SynchronizedBatchNorm2d(out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch5_conv = nn.Conv2d(in_channels,out_channels,1,1,padding=0)
        # self.branch5_bn = SynchronizedBatchNorm2d(out_channels)
        self.branch5_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv_cat = nn.Sequential(
            nn.Conv2d(self.inter,out_channels,1,1,padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        _,_,h,w = x.size()
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        x = torch.mean(x, 2, True)
        x = torch.mean(x, 3, True)
        x = self.branch5_conv(x)
        x = self.branch5_bn(x)
        x = self.relu(x)
        x = F.interpolate(x,(h,w),mode='bilinear')
        x = torch.cat([x1,x2,x3,x4,x],dim=1)
        return self.conv_cat(x)



class DPCBN(nn.Module):
    def __init__(self,in_channels,out_channels,k_size,stride = 1,padding = 0,dilation = 1):
        super(DPCBN,self).__init__()
        '''
        深度可分离Conv
        '''

        self.depthwise = nn.Conv2d(in_channels,in_channels,kernel_size=k_size,stride=stride,padding=padding,dilation=dilation,groups=in_channels,bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pointwise = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,groups=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, strides=1, atrous=None, grow_first=True):
        super(Block, self).__init__()
        if atrous == None:
            atrous = [1] * 3
        elif isinstance(atrous, int):
            atrous_list = [atrous] * 3
            atrous = atrous_list
        idx = 0
        self.head_relu = True
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
            self.head_relu = False
        else:
            self.skip = None

        self.hook_layer = None
        if grow_first:
            filters = out_filters
        else:
            filters = in_filters
        self.sepconv1 = DPCBN(in_filters, filters, 3, stride=1, padding=1 * atrous[0], dilation=atrous[0])
        self.sepconv2 = DPCBN(filters, out_filters, 3, stride=1, padding=1 * atrous[1], dilation=atrous[1])
        self.sepconv3 = DPCBN(out_filters, out_filters, 3, stride=strides, padding=1 * atrous[2],dilation=atrous[2])

    def forward(self, inp):

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x = self.sepconv1(inp)
        x = self.sepconv2(x)
        self.hook_layer = x
        x = self.sepconv3(x)

        x += skip
        return x


class Xception(nn.Module):


    def __init__(self, os):
        super(Xception, self).__init__()

        stride_list = None
        if os == 8:
            stride_list = [2, 1, 1]
        elif os == 16:
            stride_list = [2, 2, 1]
        else:
            raise ValueError('xception.py: output stride=%d is not supported.' % os)
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here

        self.block1 = Block(64, 128, 2)
        self.block2 = Block(128, 256, stride_list[0])
        self.block3 = Block(256, 728, stride_list[1])

        rate = 16 // os
        self.block4 = Block(728, 728, 1, atrous=rate)
        self.block5 = Block(728, 728, 1, atrous=rate)
        self.block6 = Block(728, 728, 1, atrous=rate)
        self.block7 = Block(728, 728, 1, atrous=rate)

        self.block8 = Block(728, 728, 1, atrous=rate)
        self.block9 = Block(728, 728, 1, atrous=rate)
        self.block10 = Block(728, 728, 1, atrous=rate)
        self.block11 = Block(728, 728, 1, atrous=rate)

        self.block12 = Block(728, 728, 1, atrous=rate)
        self.block13 = Block(728, 728, 1, atrous=rate)
        self.block14 = Block(728, 728, 1, atrous=rate)
        self.block15 = Block(728, 728, 1, atrous=rate)

        self.block16 = Block(728, 728, 1, atrous=[1 * rate, 1 * rate, 1 * rate])
        self.block17 = Block(728, 728, 1, atrous=[1 * rate, 1 * rate, 1 * rate])
        self.block18 = Block(728, 728, 1, atrous=[1 * rate, 1 * rate, 1 * rate])
        self.block19 = Block(728, 728, 1, atrous=[1 * rate, 1 * rate, 1 * rate])

        self.block20 = Block(728, 1024, stride_list[2], atrous=rate, grow_first=False)

        self.conv3 = DPCBN(1024, 1536, 3, 1, 1 * rate, dilation=rate)

        self.conv4 = DPCBN(1536, 1536, 3, 1, 1 * rate, dilation=rate)

        self.conv5 = DPCBN(1536, 2048, 3, 1, 1 * rate, dilation=rate)
        self.layers = []

        # ------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # -----------------------------

    def forward(self, input):
        self.layers = []
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        # self.layers.append(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        self.layers.append(self.block2.hook_layer)
        x = self.block3(x)
        # self.layers.append(self.block3.hook_layer)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)
        x = self.block17(x)
        x = self.block18(x)
        x = self.block19(x)
        x = self.block20(x)
        # self.layers.append(self.block20.hook_layer)

        x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu(x)

        x = self.conv4(x)
        # x = self.bn4(x)
        # x = self.relu(x)

        x = self.conv5(x)
        # x = self.bn5(x)
        # x = self.relu(x)
        self.layers.append(x)

        return x

    def get_layers(self):
        return self.layers


def get_xinception(d):

    model = Xception(d)
    return model

if __name__ == '__main__':

    load = torch.load('C:\\Users\\ZD\\Desktop\\deeplabv3plus-pytorch-master\\deeplabv3plus_xception_VOC2012_epoch46_all.pth', 'cpu')
    for i, j in enumerate(load):
        print(j)
import torch
import torch.nn as nn
NUM_ANCHORS = [6,6,6,6,4,4]  #23240

CLASS = 21
OUT_C = [1024,512] + [256] * 3


#conv_bn_relu
class BasicConv(nn.Module):

    def __init__(self,in_channels,out_channels,k_size,strides = 1,padding = 0,dilation = 1,is_relu = True,is_bn = True,is_bias = False,groups = 1):
        super(BasicConv, self).__init__()
        self.is_relu = is_relu
        self.is_bn = is_bn
        self.bias = is_bias
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=k_size,padding=padding,stride=strides,dilation=dilation,groups=groups,bias=self.bias)
        self.bn = nn.BatchNorm2d(out_channels,eps=1e-5,momentum=1e-2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.is_bn:
            x = self.bn(x)
        if self.is_relu:
            x = self.relu(x)
        return x

class BasicRFB(nn.Module):

    def __init__(self,in_channels,out_channels,strides = 1,scale = 0.1,padding = 1):
        super(BasicRFB,self).__init__()
        self.scale = scale
        inter = in_channels // 8
        self.branch0 = nn.Sequential(
            BasicConv(in_channels,2 * inter,1,strides),
            BasicConv(2 * inter,2 * inter,3,1,padding= padding,is_relu=False,dilation=padding)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_channels,inter,1),
            BasicConv(inter,2 * inter,k_size=(3,3),strides=strides,padding=(1,1)),
            BasicConv(inter * 2,inter * 2,3,padding=padding + 1,dilation=padding + 1,is_relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_channels,inter,1),
            #4(a)中5 x 5 conv拆分成两个3 x 3 conv
            BasicConv(inter,(inter // 2) * 3,3,padding=1),
            BasicConv((inter // 2) * 3,2 * inter,3,strides=strides,padding=1),
            BasicConv(2 * inter,2 * inter,3,1,padding=2 * padding + 1,dilation=2 * padding + 1,is_relu=False)
        )

        self.ConvLinear = BasicConv(6 * inter,out_channels,1,is_relu=False)
        self.shortcut = BasicConv(in_channels,out_channels,1,strides=strides,is_relu=False)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        net = torch.cat((x0,x1,x2),dim=1)
        net = self.ConvLinear(net)
        short = self.shortcut(x)
        net =  net * self.scale + short
        net = self.relu(net)

        return net


class BasicRFB_a(nn.Module):  #在论文中RFB-S只用在conv4_3

    def __init__(self, in_channels, out_channels, strides=1, scale=0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        inter = in_channels // 4

        self.branch0 = nn.Sequential(
            BasicConv(in_channels, inter, 1, strides),
            BasicConv(inter, inter, 3, 1, padding=1, is_relu=False)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_channels, inter, 1),
            BasicConv(inter, inter, k_size=(3,1), strides=1, padding=(1,0)),
            BasicConv(inter, inter, 3, padding= 3, dilation=3, is_relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_channels, inter, 1),
            BasicConv(inter, inter, (1,3),strides=strides, padding=(0,1)),
            BasicConv(inter, inter, 3, 1, padding=3, dilation=3, is_relu=False)
        )
        self.branch3 = nn.Sequential(
            BasicConv(in_channels,inter // 2,k_size=1),
            BasicConv(inter // 2,(inter // 4) * 3,k_size=(1,3),strides=1,padding=(0,1)),
            BasicConv((inter // 4) * 3,inter,k_size=(3,1),strides=strides,padding=(1,0)),
            BasicConv(inter,inter,k_size=3,strides=1,padding=5,dilation=5,is_relu=False)
        )

        self.ConvLinear = BasicConv(4 * inter, out_channels, 1, is_relu=False)
        self.shortcut = BasicConv(in_channels, out_channels, 1, strides=strides, is_relu=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.branch0(x)
        x2 = self.branch1(x)
        x3 = self.branch2(x)
        x4 = self.branch3(x)

        net = torch.cat((x1, x2, x3,x4), dim=1)
        net = self.ConvLinear(net)
        short = self.shortcut(x)
        net = net * self.scale + short
        net = self.relu(net)

        return net

def build_vgg():

    layers = []
    #conv1
    layers += [nn.Conv2d(3,64,kernel_size=3,padding=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(64,64,kernel_size=3,padding=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(kernel_size=2,stride=2)]
    #conv2
    layers += [nn.Conv2d(64,128,kernel_size=3,padding=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(128,128,kernel_size=3,padding=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(2,2)]
    #conv3
    layers += [nn.Conv2d(128,256,3,padding=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(256,256,3,padding=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(256,256,3,padding=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(2,2,ceil_mode=True)]
    #conv4
    layers += [nn.Conv2d(256, 512, 3, padding=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(512, 512, 3, padding=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(512, 512, 3, padding=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(2, 2)]
    #conv5
    layers += [nn.Conv2d(512, 512, 3, padding=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(512, 512, 3, padding=1)]
    layers += [nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(512, 512, 3, padding=1)]
    layers += [nn.ReLU(inplace=True)]

    pool5 = nn.MaxPool2d(3,1,1)
    conv6 = nn.Conv2d(512,1024,3,padding=6,dilation=6)
    conv7 = nn.Conv2d(1024,1024,1)
    layers += [pool5,conv6,nn.ReLU(inplace=True),conv7,nn.ReLU(inplace=True)]

    return layers


#conv10---
def add_RBF():
    layers = []
    layers += [BasicRFB(1024,1024,scale = 1.0,padding=2)]
    layers += [BasicRFB(1024,512,strides=2,scale=1.0,padding=2)]
    layers += [BasicRFB(512,256,scale=1.0,strides=2,padding=2)]


    layers += [BasicConv(256,128,k_size=1,strides=1)]
    layers += [BasicConv(128,256,k_size=3,strides=1)]
    layers += [BasicConv(256,128,k_size=1,strides=1)]
    layers += [BasicConv(128,256,k_size=3,strides=1)]

    return layers


def pred_layers():
    vgg_layers = build_vgg()
    rfb_a_layers = add_RBF()
    cls = []
    loc = []

    # conv4-3 pred 38x38
    loc += [nn.Conv2d(512,NUM_ANCHORS[0] * 4,kernel_size=3,padding=1)]
    cls += [nn.Conv2d(512,NUM_ANCHORS[0] * CLASS,kernel_size=3,padding=1)]

    #conv7-11 和ssd一样
    i = 1


    for k,j in enumerate(rfb_a_layers):
        if k < 3 or k % 2 == 0:
            loc += [nn.Conv2d(OUT_C[i - 1],NUM_ANCHORS[i] * 4,kernel_size=3,padding=1)]
            cls += [nn.Conv2d(OUT_C[i - 1],NUM_ANCHORS[i] * CLASS,kernel_size=3,padding=1)]

            i += 1

    return vgg_layers,rfb_a_layers,(loc,cls)





import cv2
import numpy as np

backbone,rfb_a,head = pred_layers()

class RFBNet_work(nn.Module):
    def __init__(self,is_train = False):

        super(RFBNet_work,self).__init__()
        self.is_train = is_train
        self.CLASS = 21
        self.img_size = 300
        self.base = nn.ModuleList(backbone)
        self.extras = nn.ModuleList(rfb_a)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])
        self.softmax = nn.Softmax(-1)
        self.Norm = BasicRFB_a(512,512,scale = 1.0)

    def forward(self, x):
        loc = []
        cls = []
        sources = []

        for i in range(23):
            x = self.base[i](x)

        conv4_3 = self.Norm(x)

        sources.append(conv4_3)

        for i in range(23,len(self.base)):
            x = self.base[i](x)


        for i,j in enumerate(self.extras):

            x = j(x)

            if i < 3 or i % 2 == 0:
                sources.append(x)

        for (x,l,c) in zip(sources,self.loc,self.conf):
            loc.append(l(x).permute(0,2,3,1).contiguous())
            cls.append(c(x).permute(0,2,3,1).contiguous())

        loc = torch.cat([i.view(i.size(0),-1) for i in loc],dim=1)
        cls = torch.cat([i.view(i.size(0),-1) for i in cls],dim=1)

        if self.is_train:
            loc = loc.view(loc.size(0),-1,4)
            cls = self.softmax(cls)
            cls = cls.view(cls.size(0),-1,21)
            return loc,cls
        else:
            out = (loc.view(loc.size(0),-1,4),self.softmax(cls.view(-1,self.CLASS)))

        return out

if __name__ == '__main__':
    img = cv2.imread('../3.jpg')
    img = cv2.resize(img, (300, 300))
    # img = img.transpose((2,0,1))
    # img = np.expand_dims(img, 0).transpose((0,3,1,2))
    imgs = [img] * 2
    # img = torch.from_numpy(img)
    imgs = np.array(imgs).transpose((0,3,1,2))

    imgs = torch.Tensor(imgs)
    print(imgs.shape)

    state_dict = torch.load('C:\\Users\\ZD\\Desktop\\RFBNet-master_Chinese_note-master\\RFBNet300_VOC_80_7.pth',map_location='cpu') # 读训练好的模型

    RFB = RFBNet_work()

    RFB.load_state_dict(state_dict)

    loc,cls = RFB(imgs)








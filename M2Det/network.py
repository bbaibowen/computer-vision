import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2


IMG_SIZE = 512

class Conv_Bn_Relu(nn.Module):
    def __init__(self,in_channel,out_channel,k_size,stride = 1,padding = 0,dilation = 1,groups=1,is_relu = True,is_bn = True,bias = False):
        super(Conv_Bn_Relu,self).__init__()
        self.is_bn = is_bn
        self.is_relu = is_relu
        self.conv = nn.Conv2d(in_channel,out_channel,kernel_size=k_size,stride=stride,padding=padding,
                              dilation=dilation,groups=groups,bias=bias)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channel,eps=1e-5,momentum=1e-2,affine=True)

    def forward(self, x):

        net = self.conv(x)
        if self.is_bn:
            net = self.bn(net)
        if self.is_relu:
            net = self.relu(net)
        return net

class TUM(nn.Module):

    def __init__(self,in_channel,out_channel,first_level=True):
        # super(TUM,self).__init__()
        super(TUM,self).__init__()
        self.first_level = first_level
        self.scales = 6
        self.planes = 2 * in_channel
        self.in1 = in_channel if first_level else in_channel + out_channel
        self.layers = nn.Sequential()
        self.layers.add_module('{}'.format(len(self.layers)),Conv_Bn_Relu(self.in1,self.planes,3,2,1))
        for i in range(self.scales - 2):
            if not i == self.scales - 3:
                self.layers.add_module('{}'.format(len(self.layers)),Conv_Bn_Relu(self.planes,self.planes,3,2,1))
            else:
                self.layers.add_module('{}'.format(len(self.layers)),Conv_Bn_Relu(self.planes,self.planes,3,1,0))

        self.toplayer = nn.Sequential(Conv_Bn_Relu(self.planes,self.planes,1,1,0))

        self.latlayer = nn.Sequential()
        for i in range(self.scales - 2):
            self.latlayer.add_module('{}'.format(len(self.latlayer)),Conv_Bn_Relu(self.planes,self.planes,3,1,1))

        self.latlayer.add_module('{}'.format(len(self.latlayer)),Conv_Bn_Relu(self.in1,self.planes,3,1,1))


        #1x1
        smooth = []
        for i in range(self.scales - 1):
            smooth.append(Conv_Bn_Relu(self.planes,self.planes,1,1,0))
        self.smooth = nn.Sequential(*smooth)

    def upsampling(self,feat1,feat2):
        _,_,h,w = feat2.size()

        return F.interpolate(feat1,size=(h,w)) + feat2

    def forward(self, feat1,feat2):
        if not self.first_level:
            feat1 = torch.cat([feat1,feat2],dim=1)

        conv_feat = [feat1]

        for i in range(len(self.layers)):
            feat1 = self.layers[i](feat1)
            conv_feat.append(feat1)

        deconv = [self.toplayer[0](conv_feat[-1])]

        for i in range(len(self.latlayer)):
            deconv.append(self.upsampling(deconv[i],self.latlayer[i](conv_feat[len(self.layers) - 1 - i])))

        smooth = [deconv[0]]
        for i in range(len(self.smooth)):
            smooth.append(self.smooth[i](deconv[i + 1]))

        return smooth


class SFAM(nn.Module): #融合
    def __init__(self, planes, num_levels, num_scales, compress_ratio=16):
        super(SFAM, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.ModuleList([nn.Conv2d(planes * num_levels,planes * num_levels // 16,1,1,0)] * num_scales)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.ModuleList([nn.Conv2d(planes * num_levels // 16,planes * num_levels,1,1,0)] * num_scales)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        feats = []
        for i,j in enumerate(x):
            net = self.avgpool(j)
            net = self.fc1[i](net)
            net = self.relu(net)
            net = self.fc2[i](net)
            net = self.sigmoid(net)
            feats.append(j * net)
        return feats


def vgg():
    layers = []
    layers += [nn.Conv2d(3,64,kernel_size=3,padding=1),nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(64,64,kernel_size=3,padding=1),nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(kernel_size=2,stride=2)]

    layers += [nn.Conv2d(64,128,kernel_size=3,padding=1),nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(128,128,kernel_size=3,padding=1),nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(kernel_size=2,stride=2)]

    layers += [nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]

    layers += [nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

    layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
    layers += [nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True)]

    pool5 = nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
    conv6 = nn.Conv2d(512,1024,kernel_size=3,padding=6,dilation=6)
    conv7 = nn.Conv2d(1024,1024,kernel_size=1)

    layers += [pool5,conv6,nn.ReLU(inplace=True),conv7,nn.ReLU(inplace=True)]

    return layers



class M2det(nn.Module):

    def __init__(self):
        super(M2det,self).__init__()
        self.num_level = 8

        #TUM
        for i in range(8):
            if i == 0:
                setattr(self,'unet{}'.format(i + 1),
                        TUM(in_channel = 256 // 2,out_channel = 512,first_level=True))
            else:
                setattr(self,'unet{}'.format(i + 1),TUM(in_channel = 256 // 2,out_channel = 256,first_level=False))

        self.base = nn.ModuleList(vgg())
        print(len(self.base))
        self.reduce = Conv_Bn_Relu(512, 256, k_size=3, stride=1, padding=1)
        self.up_reduce = Conv_Bn_Relu(1024, 512, k_size=1, stride=1)
        self.softmax = nn.Softmax()
        self.Norm = nn.BatchNorm2d(256 * 8)
        self.leach = nn.ModuleList([Conv_Bn_Relu(512 + 256,256 // 2,k_size=(1,1),stride=(1,1))] * 8)
        loc_ = []
        conf_ = []
        for i in range(6):#scale
            loc_.append(nn.Conv2d(256 * 8,4 * 6,3,1,1))
            conf_.append(nn.Conv2d(256 * 8,81 * 6,3,1,1))
        self.loc = nn.ModuleList(loc_)
        self.conf = nn.ModuleList(conf_)

        # self.sfam_module = SFAM(256,self.num_level,6)

    def forward(self, x):
        loc = []
        conf = []
        feats = []
        for i in range(len(self.base)):
            x = self.base[i](x)
            if i in [22,34]:
                feats.append(x)


        meger_f1 = self.reduce(feats[0])
        meger_f2 = F.interpolate(self.up_reduce(feats[1]),scale_factor=2)
        feats = torch.cat([meger_f1,meger_f2],dim=1)



        #tum
        tum_layers = [getattr(self,'unet{}'.format(1))(self.leach[0](feats),'none')]
        print('111',tum_layers[0][-1].shape)

        for i in range(1,8):
            tum_layers.append(
                getattr(self,'unet{}'.format(i + 1))(self.leach[i](feats),tum_layers[i - 1][-1])
            )
        scale_feats = [
            torch.cat([_fx[i - 1] for _fx in tum_layers],1) for i in range(6,0,-1)
        ]

        scale_feats[0] = self.Norm(scale_feats[0])
        for i in scale_feats:
            print(i.shape)



        for (x,l,c) in zip(scale_feats,self.loc,self.conf):
            loc.append(l(x).permute(0,2,3,1).contiguous())
            conf.append(c(x).permute(0,2,3,1).contiguous())

        loc = torch.cat([o.view(o.size(0),-1) for o in loc],dim=1)
        conf = torch.cat([o.view(o.size(0),-1) for o in conf],dim=1)

        loc = loc.view(loc.size(0),-1,4)
        conf = self.softmax(conf.view(-1,81))

        return loc,conf,scale_feats

class Model(nn.Module):

    def __init__(self):
        super(Model,self).__init__()
        self.module = M2det()

    def forward(self, x):
        return self.module(x)

if __name__ == '__main__':
    # load = torch.load('m2det512_vgg.pth',map_location='cpu')
    # for i in load:
    #     print(i)
    # for i,j in enumerate(load):
    #     print(i,j)
    # print('------------------------------------------------------------------------')
    # paths = ['../3.jpg','../road.jpg']
    imgs = np.array(cv2.resize(cv2.imread('../3.jpg'),(512,512)))
    imgs = np.expand_dims(imgs,0)
    imgs = np.transpose(imgs,(0,3,1,2))

    network = Model()
    # network.load_state_dict(load)
    # for i,j in network.named_parameters():
    #     print(i)
    x = torch.FloatTensor(imgs)
    loc,cls,sc = network(x)

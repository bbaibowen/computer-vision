from nnet import get_xinception,ASPP
import torch.nn as nn
import torch

RATES = [1,6,12,18]


class DeepLab3(nn.Module):


    def __init__(self,rates):
        super(DeepLab3,self).__init__()
        self.inter = 256
        self.backbone = get_xinception(16)
        self.aspp = ASPP(2048,self.inter,rates)
        self.dropout1 = nn.Dropout(0.5)
        self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=4)
        self.shortcut_conv = nn.Sequential(
            nn.Conv2d(self.inter, 48, 1, 1,0),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.cat_conv = nn.Sequential(
            nn.Conv2d(self.inter + 48, self.inter, 3, 1, padding=1),
            nn.BatchNorm2d(self.inter),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(self.inter, self.inter, 3, 1, padding=1),
            nn.BatchNorm2d(self.inter),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.cls_conv = nn.Conv2d(self.inter, 21, 1, 1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.backbone_layers = self.backbone.get_layers()



    def forward(self, x):
        _ = self.backbone(x)
        layers = self.backbone.get_layers()
        feature_aspp = self.aspp(layers[-1])
        feature_aspp = self.dropout1(feature_aspp)
        feature_aspp = self.upsample_sub(feature_aspp)
        feature_shallow = self.shortcut_conv(layers[0])
        feature_cat = torch.cat([feature_aspp, feature_shallow], 1)
        result = self.cat_conv(feature_cat)
        result = self.cls_conv(result)
        result = self.upsample4(result)
        return result


def get_deeplab3(PATH = None,rates = RATES,train = True):

    if train:
        return DeepLab3(rates).train()
    else:
        net = nn.DataParallel(DeepLab3(rates))
        if PATH is not None:
            net.load_state_dict(torch.load(
                PATH,
                'cpu'))
        return net




if __name__ == '__main__':
    x = torch.randn(1,3,512,512)
    # net = get_deeplab3('C:\\Users\\ZD\\Desktop\\deeplabv3plus-pytorch-master\\deeplabv3plus_xception_VOC2012_epoch46_all.pth')    y = net(x)
    # print(y.shape)

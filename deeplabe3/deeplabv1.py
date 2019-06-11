import torch.nn as nn
import torch
from resnet import resnet50


class DeepLabv1(nn.Module):


    def __init__(self,num_cls = 21):

        super(DeepLabv1,self).__init__()
        self.resnet = resnet50()
        self.cls = nn.Conv2d(2048,num_cls,1)

    def forward(self,x):
        x = self.resnet(x)
        x = self.cls(x)

        print(x.shape)
        return x

if __name__ == '__main__':
    model = DeepLabv1()
    x = torch.randn(1,3,512,512)
    y = model(x)
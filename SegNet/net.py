from torch import nn
from torch.nn import functional as F
import torch

class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()
        self.class_num = 21
        #Encoder
        self.vgg16_block = [2,2,3,3,3]
        self.vgg16_kernel_num = [3,64,128,256,512,512]
        for i in range(len(self.vgg16_block)):
            block=[]
            for j in range(self.vgg16_block[i]):
                if j == 0:
                    block.append(nn.Conv2d(self.vgg16_kernel_num[i],self.vgg16_kernel_num[i+1],kernel_size=3,
                                           padding=1))
                else:
                    block.append(nn.Conv2d(self.vgg16_kernel_num[i+1],self.vgg16_kernel_num[i+1],kernel_size=3,
                                           padding=1))
                block.append((nn.BatchNorm2d(self.vgg16_kernel_num[i+1])))
                block.append(nn.ReLU(inplace=True))
            block.append(nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True))
            setattr(self, 'vgg16_block' + str(i + 1), nn.Sequential(*block))

        #Decoder
        self.decoder_block = [3, 3, 3, 2, 2]
        self.decoder_kernel_num = [512, 512, 256, 128, 64, self.class_num]
        for i in range(len(self.decoder_block)):
            block = []
            for j in range(self.decoder_block[i]):
                if j != self.decoder_block[i]-1:
                    block.append(nn.Conv2d(self.decoder_kernel_num[i], self.decoder_kernel_num[i], kernel_size=3,
                                           padding=1))
                    block.append((nn.BatchNorm2d(self.decoder_kernel_num[i])))
                else:
                    block.append(nn.Conv2d(self.decoder_kernel_num[i], self.decoder_kernel_num[i + 1], kernel_size=3,
                                           padding=1))
                    block.append((nn.BatchNorm2d(self.decoder_kernel_num[i+1])))
                block.append(nn.ReLU(inplace=True))
            setattr(self, 'decoder_block' + str(len(self.decoder_block)-i), nn.Sequential(*block))

    def forward(self, x):
        x_size = x.size()
        xe1, idx1 = self.vgg16_block1(x)
        xe1_size = xe1.size()
        xe2, idx2 = self.vgg16_block2(xe1)
        xe2_size = xe2.size()
        xe3, idx3 = self.vgg16_block3(xe2)
        xe3_size = xe3.size()
        xe4, idx4 = self.vgg16_block4(xe3)
        xe4_size = xe4.size()
        xe5, idx5 = self.vgg16_block5(xe4)
        xd5 = F.max_unpool2d(xe5, idx5, kernel_size=2, stride=2, output_size=xe4_size)
        xd5 = self.decoder_block5(xd5)
        xd4 = F.max_unpool2d(xd5, idx4, kernel_size=2, stride=2, output_size=xe3_size)
        xd4 = self.decoder_block4(xd4)
        xd3 = F.max_unpool2d(xd4, idx3, kernel_size=2, stride=2, output_size=xe2_size)
        xd3 = self.decoder_block3(xd3)
        xd2 = F.max_unpool2d(xd3, idx2, kernel_size=2, stride=2, output_size=xe1_size)
        xd2 = self.decoder_block2(xd2)
        xd1 = F.max_unpool2d(xd2, idx1, kernel_size=2, stride=2, output_size=x_size)
        xd1 = self.decoder_block1(xd1)

        return xd1

if __name__ == '__main__':
    x = torch.randn(1,3,500,500)
    model = SegNet()
    y = model(x)
    print(y.shape)
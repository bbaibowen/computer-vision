import torch
import torch.nn as nn

class GN(nn.Module):

    def __init__(self,out_channels,groups):
        super(GN,self).__init__()

        self.out_channels = out_channels
        self.num_groups = groups
        self.dim_per = -1
        self.eps = 1e-5

        self.gn = nn.GroupNorm(self.num_groups,self.out_channels,self.eps,True)

    def forward(self,x):
        return self.gn(x)


import torch
import torch.nn as nn
import numpy as np
# from network import *
from torch.autograd import Function

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

def comp(a,b,A,B):
    batch = a.size(0)
    a_ = a.unsqueeze(1).contiguous().view(batch,1,-1)
    b_ = b.unsqueeze(1).contiguous().view(batch,1,-1)
    c_ = torch.cat((a_,b_),1)
    m = c_.max(1)[0].unsqueeze(1).expand_as(c_)
    m = (c_==m).float()
    m1 = m.permute(0,2,1)
    k = m1[...,0]
    j = m1[...,1]
    z = ((k*j)!=1).float()
    j = z*j
    m1 = torch.cat((k,j),1).unsqueeze(1).view_as(m)

    A_ = A.unsqueeze(1).contiguous().view(batch,1,-1)
    B_ = B.unsqueeze(1).contiguous().view(batch,1,-1)
    C_ = torch.cat((A_,B_),1).permute(0,2,1)
    m1 = m1.long().permute(0,2,1)
    res = C_[m1.long()==1].view_as(a)

    return res


class left_pooling(Function):

    def forward(self, input_):
        self.save_for_backward(input_.clone())
        output = torch.zeros_like(input_)
        batch = input_.size(0)
        width = input_.size(3)


        input_tmp = input_.select(3, width - 1)
        output.select(3, width - 1).copy_(input_tmp)

        for idx in range(1, width):
            input_tmp = input_.select(3, width - idx - 1)
            output_tmp = output.select(3, width - idx)
            cmp_tmp = torch.cat((input_tmp.view(batch, 1, -1), output_tmp.view(batch, 1, -1)), 1).max(1)[0]
            output.select(3, width - idx - 1).copy_(cmp_tmp.view_as(input_tmp))

        return output

    def backward(self, grad_output):
        input_, = self.saved_tensors
        output = torch.zeros_like(input_)

        grad_output = grad_output.clone()
        res = torch.zeros_like(grad_output)

        w = input_.size(3)
        batch = input_.size(0)

        output_tmp = res.select(3, w - 1)
        grad_output_tmp = grad_output.select(3, w - 1)
        output_tmp.copy_(grad_output_tmp)

        input_tmp = input_.select(3, w - 1)
        output.select(3, w - 1).copy_(input_tmp)

        for idx in range(1, w):
            input_tmp = input_.select(3, w - idx - 1)
            output_tmp = output.select(3, w - idx)
            cmp_tmp = torch.cat((input_tmp.view(batch, 1, -1), output_tmp.view(batch, 1, -1)), 1).max(1)[0]
            output.select(3, w - idx - 1).copy_(cmp_tmp.view_as(input_tmp))

            grad_output_tmp = grad_output.select(3, w - idx - 1)
            res_tmp = res.select(3, w - idx)
            com_tmp = comp(input_tmp, output_tmp, grad_output_tmp, res_tmp)
            res.select(3, w - idx - 1).copy_(com_tmp)
        return res


class right_pooling(Function):
    def forward(self, input_):
        self.save_for_backward(input_)

        output = torch.zeros_like(input_)
        width = input_.size(3)
        batch = input_.size(0)

        input_tmp = input_.select(3, 0)
        output.select(3, 0).copy_(input_tmp)

        for idx in range(1, width):
            input_tmp = input_.select(3, idx)
            output_tmp = output.select(3, idx - 1)

            cmp_tmp = torch.cat((input_tmp.view(batch, 1, -1), output_tmp.view(batch, 1, -1)), 1).max(1)[0]
            output.select(3, idx).copy_(cmp_tmp.view_as(input_tmp))
        return output

    def backward(self, grad_output):
        input_, = self.saved_tensors
        output = torch.zeros_like(input_)

        grad_output = grad_output.clone()
        res = torch.zeros_like(grad_output)

        w = input_.size(3)
        batch = input_.size(0)

        output_tmp = res.select(3, 0)
        grad_output_tmp = grad_output.select(3, 0)
        output_tmp.copy_(grad_output_tmp)

        input_tmp = input_.select(3, 0)
        output.select(3, 0).copy_(input_tmp)

        for idx in range(1, w):
            input_tmp = input_.select(3, idx)
            output_tmp = output.select(3, idx - 1)
            cmp_tmp = torch.cat((input_tmp.view(batch, 1, -1), output_tmp.view(batch, 1, -1)), 1).max(1)[0]
            output.select(3, idx).copy_(cmp_tmp.view_as(input_tmp))

            grad_output_tmp = grad_output.select(3, idx)
            res_tmp = res.select(3, idx - 1)
            com_tmp = comp(input_tmp, output_tmp, grad_output_tmp, res_tmp)
            res.select(3, idx).copy_(com_tmp)
        return res


class top_pooling(Function):

    def forward(self, input_):
        self.save_for_backward(input_)
        output = torch.zeros_like(input_)
        height = output.size(2)
        batch = input_.size(0)

        input_tmp = input_.select(2, height - 1)
        output.select(2, height - 1).copy_(input_tmp)

        for idx in range(1, height):
            input_tmp = input_.select(2, height - idx - 1)
            output_tmp = output.select(2, height - idx)
            cmp_tmp = \
            torch.cat((input_tmp.contiguous().view(batch, 1, -1), output_tmp.contiguous().view(batch, 1, -1)), 1).max(
                1)[0]

            output.select(2, height - idx - 1).copy_(cmp_tmp.view_as(input_tmp))
        return output

    def backward(self, grad_output):
        input_, = self.saved_tensors
        output = torch.zeros_like(input_)

        grad_output = grad_output.clone()
        res = torch.zeros_like(grad_output)

        height = output.size(2)
        batch = input_.size(0)
        # copy the last row
        input_tmp = input_.select(2, height - 1)
        output.select(2, height - 1).copy_(input_tmp)

        grad_tmp = grad_output.select(2, height - 1)
        res.select(2, height - 1).copy_(grad_tmp)
        for idx in range(1, height):
            input_tmp = input_.select(2, height - idx - 1)
            output_tmp = output.select(2, height - idx)
            cmp_tmp = \
            torch.cat((input_tmp.contiguous().view(batch, 1, -1), output_tmp.contiguous().view(batch, 1, -1)), 1).max(
                1)[0]
            output.select(2, height - idx - 1).copy_(cmp_tmp.view_as(input_tmp))

            grad_output_tmp = grad_output.select(2, height - idx - 1)
            res_tmp = res.select(2, height - idx)

            com_tmp = comp(input_tmp, output_tmp, grad_output_tmp, res_tmp)
            res.select(2, height - idx - 1).copy_(com_tmp)
        return res


class bottom_pooling(Function):
    def forward(self, input_):
        self.save_for_backward(input_)
        output = torch.zeros_like(input_)
        height = output.size(2)
        batch = output.size(0)

        input_tmp = input_.select(2, 0)
        output.select(2, 0).copy_(input_tmp)

        for idx in range(1, height):
            input_tmp = input_.select(2, idx)
            output_tmp = output.select(2, idx - 1)
            cmp_tmp = \
            torch.cat((input_tmp.contiguous().view(batch, 1, -1), output_tmp.contiguous().view(batch, 1, -1)), 1).max(
                1)[0]
            output.select(2, idx).copy_(cmp_tmp.view_as(input_tmp))
        return output

    def backward(self, grad_output):
        input_, = self.saved_tensors
        output = torch.zeros_like(input_)

        grad_output = grad_output.clone()
        res = torch.zeros_like(grad_output)

        height = output.size(2)
        batch = output.size(0)

        input_tmp = input_.select(2, 0)
        output.select(2, 0).copy_(input_tmp)

        grad_tmp = grad_output.select(2, 0)
        res.select(2, 0).copy_(grad_tmp)

        for idx in range(1, height):
            input_tmp = input_.select(2, idx)
            output_tmp = output.select(2, idx - 1)
            cmp_tmp = \
            torch.cat((input_tmp.contiguous().view(batch, 1, -1), output_tmp.contiguous().view(batch, 1, -1)), 1).max(
                1)[0]
            output.select(2, idx).copy_(cmp_tmp.view_as(input_tmp))

            grad_output_tmp = grad_output.select(2, idx)
            res_tmp = res.select(2, idx - 1)

            com_tmp = comp(input_tmp, output_tmp, grad_output_tmp, res_tmp)
            res.select(2, idx).copy_(com_tmp)
        return res

class pool(nn.Module):

    def __init__(self,channels,pool1,pool2):
        super(pool,self).__init__()
        self.p1_conv1 = convolution(3,channels,128)
        self.p2_conv1 = convolution(3,channels,128)
        self.p_conv1 = nn.Conv2d(128,channels,(3,3),padding=(1,1),bias=False)
        self.p_bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels,channels,(1,1),bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = convolution(3,channels,channels)
        self.pool1 = pool1()
        self.pool2 = pool2()

    def forward(self, x):
        p1_conv1 = self.p1_conv1(x)
        pool1 = self.pool1(p1_conv1)


        p2_conv1 = self.p2_conv1(x)
        pool2 = self.pool2(p2_conv1)  


        p_conv1 = self.p_conv1(pool1 + pool2)
        p_bn1 = self.p_bn1(p_conv1)

        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu(p_bn1 + bn1)

        conv2 = self.conv2(relu1)
        return conv2


def get_tl_pool(channels):

    return pool(channels=channels,pool1=top_pooling,pool2=left_pooling)

def get_br_pool(channels):

    return pool(channels=channels,pool1=bottom_pooling,pool2=right_pooling)




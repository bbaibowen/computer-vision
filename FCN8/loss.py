import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):

    def __init__(self):
        super(Loss,self).__init__()


    def forward(self, pred,label):

        batch,channel,h,w = pred.size()
        pred = pred.view(-1,channel)
        label = label.view(-1)
        loss = nn.CrossEntropyLoss(size_average=False,ignore_index=255)(pred,label)
        return loss
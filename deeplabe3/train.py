import numpy as np
import torch.utils.data as data
from network import get_deeplab3
from data_load import VOC_load
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

'''
简单写写
'''

batch = 2
datasets = VOC_load(year=2012, root='../des/VOCdevkit')
dataloader = data.DataLoader(datasets,batch_size=batch,shuffle=True)
net = get_deeplab3(PATH = None)

loss_layer = nn.CrossEntropyLoss(ignore_index=255)
op = optim.SGD(net.parameters(),lr=1e-4)
total_loss = 0
for i,(im,label) in enumerate(dataloader):
    iter_loss = 0
    im = Variable(im)
    labels = [Variable(l) for l in label]
    outs = net(im)
    op.zero_grad()
    for out,label in zip(outs,labels):
        loss = loss_layer(out.unsqueeze(0),label.unsqueeze(0))
        iter_loss += loss
    iter_loss /= batch
    print(iter_loss)
    iter_loss.backward()
    total_loss += iter_loss.item()
    op.step()

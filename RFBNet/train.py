from bulid_net import RFBNet_work
from anchors_layer import gen_anchors
import torch
import numpy as np
from load_data import VOC_load
from Loss import MultiBoxLoss
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)

USE_PRE_TRAIN = False
pre_train_path = ''
IS_TRAIN = True
MAX_ITER = 15000
BATCH = 3
ANCHORS = gen_anchors()
load_data = VOC_load()

net = RFBNet_work(True)
Loss_function = MultiBoxLoss()

if USE_PRE_TRAIN:
    net.base.load_state_dict(torch.load(pre_train_path))

op = optim.SGD(net.parameters(),lr=4e-3,momentum=0.9,weight_decay=5e-4)

#or DATA.tensordata(x=,y=)
#collate_fn：取数据的函数,这里本来可以直接用false,但是target第一个维度不唯一，不重新定义会报错
train_data = data.DataLoader(load_data,BATCH,True,collate_fn=detection_collate)

for i in range(len(load_data)):
    for i, (input, target) in enumerate(train_data):
        imgs,tt = input, target  #list
        print(len(imgs),len(tt))
        imgs = Variable(imgs)
        tt = [Variable(i) for i in tt]
        pred = net(imgs)
        op.zero_grad()
        loss_total = Loss_function(pred, ANCHORS,tt)
        loss_total.backward()
        op.step()

        print(loss_total)

# torch.save(net.state_dict(),'路径')

#关于learning rate调整：

# def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    #iteration：调整的节点（训练到多少步调整一次）
    #epoch_size：len(dataset) // args.batch_size
    # step_index：初始化0，调整一次+1
    # if epoch < 6:
    #     lr = 1e-6 + (args.lr-1e-6) * iteration / (epoch_size * 5)
    # else:
    #     lr = args.lr * (gamma ** (step_index))
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    # return lr
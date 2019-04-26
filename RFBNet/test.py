from anchors_layer import gen_anchors
import torch
import torch.nn as nn
from bulid_net import RFBNet_work
import cv2
import numpy as np
from utils import box_decode,nms
import time
import torchvision.transforms as transforms
classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]
scales = [0.1,0.2]

def handle_img(path):
    img = cv2.imread(path)
    img = cv2.resize(img,(300,300))
    img = np.array(img).astype(np.float32)
    img -= np.array((104, 117, 123))
    return img

PATH = ['../person.jpg','../road.jpg']

ANCHORS = gen_anchors().data
CLASS = 21
state_dict = torch.load('C:\\Users\\ZD\\Desktop\\RFBNet-master_Chinese_note-master\\RFBNet300_VOC_80_7.pth',
                        map_location='cpu')
net = RFBNet_work()
net.load_state_dict(state_dict)
imgs = np.array([handle_img(i) for i in PATH]).transpose((0,3,1,2))
imgs = torch.Tensor(imgs)
loc,cls = net(imgs)
cls = cls.view(-1,11620,21)

for i in range(len(PATH)):

    show_img = cv2.imread(PATH[i])
    img_shape = show_img.shape
    # show_img = cv2.resize(show_img, (300, 300))
    cls_data = cls[i,...].view(11620,21).data.numpy()
    loc_data = loc[i,...].view(11620,4).data
    # print(loc_data.shape,cls_data.shape,ANCHORS.shape)
    decode = box_decode(loc_data,ANCHORS,scales).numpy()
    boxes = []
    # NMS
    for j in range(1,21):
        index = np.where(cls_data[:,j] > 0.4)[0]
        f_boxes = decode[index]
        f_score = cls_data[index,j]
        merge = np.hstack((f_boxes,f_score[:,np.newaxis])).astype(np.float32)
        nms_index = nms(merge,0.8)
        merge = merge[nms_index,:]
        boxes.append(merge)
    for bb,box in enumerate(boxes):

        if len(box) != 0:
            for tt in box:
                print(tt)
                x1,y1,x2,y2,s = tt
                x1 *= img_shape[1]
                y1 *= img_shape[0]
                x2 *= img_shape[1]
                y2 *= img_shape[0]

                p1 = (int(x1),int(y2))
                p2 = (int(x2),int(y1))
                print(p1,p2)
                cv2.rectangle(show_img, p1, p2, (255,0,0), 1)
                text = '%s/%.3f' % (classes[bb], s)
                text_loc = (p1[1], p1[0])
                cv2.putText(show_img, text, text_loc[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,0,0), 1)
    cv2.imshow('RFB', show_img)
    cv2.waitKey(0)










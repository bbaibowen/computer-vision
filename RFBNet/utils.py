import torch
import numpy as np


def box_decode(loc,anchors,scale):

    boxes = torch.cat((
        anchors[:,:2] + loc[:,:2] * scale[0] * anchors[:,2:],
        anchors[:,2:] * torch.exp(loc[:,2:] * scale[1])
    ),dim=1)

    boxes[:,:2] -= boxes[:,2:] / 2
    boxes[:,2:] += boxes[:,:2]

    return boxes

def box_encode(box,anchor,scales):

    xy = (box[:,:2] + box[:,2:]) / 2 - anchor[:,:2]
    xy /= (scales[0] * anchor[:,2:])
    wh = (box[:,2:] - box[:,:2]) / anchor[:,2:]
    wh = torch.log(wh) / scales[1]
    encode = torch.cat([xy,wh],dim=1)
    return encode

def nms(dets, thresh = 0.5):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0] # pred bbox top_x
    y1 = dets[:, 1] # pred bbox top_y
    x2 = dets[:, 2] # pred bbox bottom_x
    y2 = dets[:, 3] # pred bbox bottom_y
    scores = dets[:, 4] # pred bbox cls score
    # pred bbox areas,
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) #注意要加1才是真正的边长像素（如x2=6,x1=2,则长度实际为5 ）
    order = scores.argsort()[::-1] # 对pred bbox按score做降序排序

    keep = [] # NMS后，保留的pred bbox
    while order.size > 0:
        i = order[0] # top-1 score bbox
        keep.append(i) # top-1 score的话，自然就保留了
        xx1 = np.maximum(x1[i], x1[order[1:]]) # top-1 bbox（score最大）与order中剩余bbox计算NMS
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)#计算IOU

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


import numpy as np
import tensorflow as tf


SEARCH = 255
SEARCH_FEAT = 17
BASE = 64
STRIDES = 16
SACLE = [1/3,1/2,1,2,3]
K = 5
NUM_ANCHORS = 17 * 17 * len(SACLE)
IOU_THRESHOLD = 0.3

def iou(box1,box2):
    N = box1.shape[0]
    K = box2.shape[0]
    box1 = np.array(box1.reshape((N, 1, 4))) + np.zeros((1, K, 4))  # box1=[N,K,4]
    box2 = np.array(box2.reshape((1, K, 4))) + np.zeros((N, 1, 4))  # box1=[N,K,4]
    x_max = np.max(np.stack((box1[:, :, 0], box2[:, :, 0]), axis=-1), axis=2)
    x_min = np.min(np.stack((box1[:, :, 2], box2[:, :, 2]), axis=-1), axis=2)
    y_max = np.max(np.stack((box1[:, :, 1], box2[:, :, 1]), axis=-1), axis=2)
    y_min = np.min(np.stack((box1[:, :, 3], box2[:, :, 3]), axis=-1), axis=2)
    tb = x_min - x_max
    lr = y_min - y_max
    tb[np.where(tb < 0)] = 0
    lr[np.where(lr < 0)] = 0
    over_square = tb * lr
    all_square = (box1[:, :, 2] - box1[:, :, 0]) * (box1[:, :, 3] - box1[:, :, 1]) + (box2[:, :, 2] - box2[:, :, 0]) * (
                box2[:, :, 3] - box2[:, :, 1]) - over_square
    return over_square / all_square


def xywh2xxyy(_anchor):
    box = np.zeros_like(_anchor)
    # x1,y1,x2,y2
    box[:, 0] = _anchor[:, 0] - _anchor[:, 2] / 2
    box[:, 1] = _anchor[:, 1] - _anchor[:, 3] / 2
    box[:, 2] = _anchor[:, 0] + _anchor[:, 2] / 2
    box[:, 3] = _anchor[:, 1] + _anchor[:, 3] / 2

    return box


def singe_anchor():

    scale = np.array(SACLE)
    s = BASE * BASE
    w = np.sqrt(s / scale)
    h = w * scale
    x = y = (STRIDES - 1) / 2
    # _anchor = [x * np.ones_like(scale),y * np.ones_like(scale),w,h]
    # _anchor = np.vstack(_anchor) #[[x,x,x,x,x],[y,y,y,y,y],.......]
    # _anchor = _anchor.T   <==> np.transpose
    _anchor = np.vstack([x * np.ones_like(scale),y * np.ones_like(scale),w,h]).T

    anchors = xywh2xxyy(_anchor).astype(np.int32)


    return anchors



def gen_anchors():
    anchors = singe_anchor()
    shift_x = [i * STRIDES for i in range(SEARCH_FEAT)]
    shift_y = [i * STRIDES for i in range(SEARCH_FEAT)]
    shift_x,shift_y = np.meshgrid(shift_x,shift_y)
    shift_me = np.vstack([shift_x.ravel(),shift_y.ravel(),shift_x.ravel(),shift_y.ravel()]).T
    anchors = anchors.reshape((1,K,4)) + shift_me.reshape((shift_me.shape[0],1,4))
    anchors = anchors.reshape((K * shift_me.shape[0],4)).astype(np.float32)

    return anchors #x1,y1,x2,y2


if __name__ == '__main__':

    #17x17x5 = 1445个
    anchors = gen_anchors()
    print(anchors,anchors.shape)
import numpy as np
import cv2

def process_img(img,scale):

    h,w,_ = img.shape
    _img = np.array(cv2.resize(img,(int(w * scale),int(h * scale))),np.float32)
    _img -= 127.5
    _img /= 128.

    return _img

#得到对应原图的box坐标，分类分数，box偏移量,pnet大致将图像size缩小2倍
def generate_box(score,box,scale,threshold):

    stride = 2
    cell_size = 12

    index = np.where(score > threshold)

    #offset
    x1,y1,x2,y2 = [box[index[0],index[1],i] for i in range(4)]
    offset = np.array([x1,x2,y1,y2])
    score = score[index[0],index[1]]

    bbox = np.vstack([
        np.round((stride * index[1]) / scale),
        np.round((stride * index[0]) / scale),
        np.round((stride * index[1] + cell_size) / scale),
        np.round((stride * index[0] + cell_size) / scale),score,offset
    ])

    return bbox.T


def NMS(dets,threshold):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 将概率值从大到小排列
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)

        # 保留小于阈值的下标，因为order[0]拿出来做比较了，所以inds+1是原来对应的下标
        inds = np.where(ovr <= threshold)[0]
        order = order[inds + 1]

    return keep


def change_box(box):
    # box:[n,5]
    #把box转成大致的一个正方形

    c_box = box.copy()

    h = box[:, 3] - box[:, 1]
    w = box[:, 2] - box[:, 0]

    # 边长
    max_s = np.maximum(w, h)

    c_box[:, 0] = box[:, 0] + w * .5 - max_s * .5
    c_box[:, 1] = box[:, 1] + h * .5 - max_s * .5
    c_box[:, 2] = c_box[:, 0] + max_s
    c_box[:, 3] = c_box[:, 1] + max_s
    return c_box


def pad(bboxes, w, h):
    '''将超出图像的box进行处理
    参数：
      bboxes:人脸框
      w,h:图像长宽
    x：候选框的左上角x坐标
    y：候选框的左上角y坐标
    ex：候选框的右下角x坐标
    ey：候选框的与下角y坐标
    dx：经过对齐之后的候选框左上角x坐标
    dy：经过对齐之后的候选框左上角y坐标
    edx：修改之后的候选框右下角x
    edy：修改之后的候选框右下角y
    tmpw：候选框的宽度
    tmph：候选框的长度
    '''
    # box的长宽
    tmpw, tmph = bboxes[:, 2] - bboxes[:, 0] + 1, bboxes[:, 3] - bboxes[:, 1] + 1
    num_box = bboxes.shape[0]

    dx, dy = np.zeros((num_box,)), np.zeros((num_box,))
    edx, edy = tmpw.copy() - 1, tmph.copy() - 1
    # box左上右下的坐标
    x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    # 找到超出右下边界的box并将ex,ey归为图像的w,h
    # edx,edy为调整后的box右下角相对原box左上角的相对坐标
    tmp_index = np.where(ex > w - 1)
    edx[tmp_index] = tmpw[tmp_index] + w - ex[tmp_index] - 2
    ex[tmp_index] = w - 1

    tmp_index = np.where(ey > h - 1)
    edy[tmp_index] = tmph[tmp_index] + h - ey[tmp_index] - 2
    ey[tmp_index] = h - 1
    # 找到超出左上角的box并将x,y归为0
    # dx,dy为调整后的box的左上角坐标相对于原box左上角的坐标
    tmp_index = np.where(x < 0)
    dx[tmp_index] = 0 - x[tmp_index]
    x[tmp_index] = 0

    tmp_index = np.where(y < 0)
    dy[tmp_index] = 0 - y[tmp_index]
    y[tmp_index] = 0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
    return_list = [item.astype(np.int32) for item in return_list]

    return return_list


def calibrate_box(bbox, reg):
    '''校准box
    参数：
      bbox:pnet生成的box

      reg:rnet生成的box偏移值
    返回值：

    '''

    bbox_c = bbox.copy()
    w = bbox[:, 2] - bbox[:, 0]
    w = np.expand_dims(w, 1)
    h = bbox[:, 3] - bbox[:, 1]
    h = np.expand_dims(h, 1)
    reg_m = np.hstack([w, h, w, h])
    aug = reg_m * reg
    bbox_c[:, 0:4] = bbox_c[:, 0:4] + aug
    return bbox_c


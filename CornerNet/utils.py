import torch.nn as nn
import torch
import numpy as np


def heats_nms(heatmap,k_size = 3):
    hmax = nn.functional.max_pool2d(heatmap, (k_size, k_size),stride=1,padding=(k_size - 1) // 2)
    # hmax = nn.MaxPool2d(heatmap,(k_size,k_size),stride=1,padding=(k_size - 1) // 2)
    keep = (hmax == heatmap).float() #[0,0,1,0,1]
    return heatmap * keep

def Top_k(scores,k):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)

    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):


    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def decode_layer(out_layers,topk = 100,ae_threshold = 0.5,num_dets = 100):

    tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr = out_layers
    num_im,channel,h,w = tl_heat.size()

    #heat
    tl_heat = torch.sigmoid(tl_heat)
    tl_heat = heats_nms(tl_heat)
    tl_scores,tl_index,tl_cls,tl_y,tl_x = Top_k(tl_heat,topk)
    br_heat = torch.sigmoid(br_heat)
    br_heat = heats_nms(br_heat)
    br_scores,br_index,br_cls,br_y,br_x = Top_k(br_heat,topk)
    tl_y = tl_y.view(num_im,topk,1).expand(num_im,topk,topk)
    tl_x = tl_x.view(num_im,topk,1).expand(num_im,topk,topk)
    br_y = br_y.view(num_im,1,topk).expand(num_im,topk,topk)
    br_x = br_x.view(num_im,1,topk).expand(num_im,topk,topk)

    #offset
    tl_regr = _tranpose_and_gather_feat(tl_regr,tl_index)
    tl_regr = tl_regr.view(num_im,topk,1,2)
    br_regr = _tranpose_and_gather_feat(br_regr, br_index)
    br_regr = br_regr.view(num_im, 1, topk, 2)
    tl_x += tl_regr[...,0]
    tl_y += tl_regr[...,1]
    br_x += br_regr[...,0]
    br_y += br_regr[...,1]
    bboxes = torch.stack((tl_x,tl_y,br_x,br_y),dim=3)


    #embedding
    tl_tag = _tranpose_and_gather_feat(tl_tag,tl_index)
    tl_tag = tl_tag.view(num_im, topk, 1)
    br_tag = _tranpose_and_gather_feat(br_tag,br_index)
    br_tag = br_tag.view(num_im,1,topk)
    distance = torch.abs(tl_tag - br_tag)
    tl_scores = tl_scores.view(num_im,topk,1).expand(num_im,topk,topk)
    br_scores = br_scores.view(num_im,1,topk).expand(num_im,topk,topk)
    scores = (tl_scores + br_scores) / 2

    tl_cls = tl_cls.view(num_im,topk,1).expand(num_im,topk,topk)
    br_cls = br_cls.view(num_im,1,topk).expand(num_im,topk,topk)
    cls_index = (tl_cls != br_cls)
    distance_index = (distance > ae_threshold)
    #zuo（x1,y2）  右(x2,y1)
    w_index = (br_x < tl_x)
    h_index = (br_y < tl_y)

    scores[cls_index] = -1
    scores[distance_index] = -1
    scores[w_index] = -1
    scores[h_index] = -1

    scores = scores.view(num_im,-1)
    scores,inds = torch.topk(scores,num_dets)
    scores = scores.unsqueeze(2)

    bboxes = bboxes.view(num_im,-1,4)
    bboxes = _gather_feat(bboxes,inds)

    clses = tl_cls.contiguous().view(num_im, -1, 1)
    clses = _gather_feat(clses,inds).float()

    tl_scores = tl_scores.contiguous().view(num_im, -1, 1)
    tl_scores = _gather_feat(tl_scores, inds).float()
    br_scores = br_scores.contiguous().view(num_im, -1, 1)
    br_scores = _gather_feat(br_scores, inds).float()

    outs = torch.cat([bboxes,scores,tl_scores,br_scores,clses],dim=2)

    return outs


def soft_nms(boxes,sigma = 0.5,Nt = 0.3,threshold = 0.1,method=0):
    batch = boxes.shape[0]


    for i in range(batch):

        maxscore = boxes[i,4]
        maxpos = i
        tx1,ty1,tx2,ty2,ts = boxes[i,0],boxes[i,1],boxes[i,2],boxes[i,3],boxes[i,4]
        pos = i + 1
        while pos < batch:
            if maxscore < boxes[pos,4]:
                maxscore = boxes[pos,4]
                maxpos = pos
            pos += 1
        boxes[i,0] = boxes[maxpos,0]
        boxes[i,1] = boxes[maxpos,1]
        boxes[i,2] = boxes[maxpos,2]
        boxes[i,3] = boxes[maxpos,3]
        boxes[i,4] = boxes[maxpos,4]

        boxes[maxpos,0] = tx1
        boxes[maxpos,1] = ty1
        boxes[maxpos,2] = tx2
        boxes[maxpos,3] = ty2
        boxes[maxpos,4] = ts

        tx1,ty1,tx2,ty2,ts = boxes[i,0],boxes[i,1],boxes[i,2],boxes[i,3],boxes[i,4]

        while pos < batch:
            x1 = boxes[pos,0]
            y1 = boxes[pos,1]
            x2 = boxes[pos,2]
            y2 = boxes[pos,3]
            s = boxes[pos,4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            w = (min(tx2,x2) - max(tx1,x1) + 1)
            if w > 0:
                h = min(ty2,y2) - max(ty1,y1) + 1
                if h > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - w * h)
                    ov = w * h / ua

                    if method == 1:  #线性
                        weight = 1 - ov if ov > Nt else 1
                    elif method == 2:  #高斯
                        weight = np.exp(-(ov * ov) / sigma)

                    boxes[pos,4] *= weight

                    if boxes[pos, 4] < threshold:
                        boxes[pos,0] = boxes[batch-1, 0]
                        boxes[pos,1] = boxes[batch-1, 1]
                        boxes[pos,2] = boxes[batch-1, 2]
                        boxes[pos,3] = boxes[batch-1, 3]
                        boxes[pos,4] = boxes[batch-1, 4]
                        batch -= 1
                        pos -= 1
            pos += 1

    return [i for i in range(batch)]


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









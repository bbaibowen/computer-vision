import torch
import torch.nn as nn
import numpy as np
import cv2
from hourglass import get_large_hourglass_net,load_weights
from utils import process,ctdet_decode,ctdet_post_process,soft_nms


PATH = 'ctdet_coco_hg.pth'
IMG_PATH = '../0002.png'
SCALES = [0.5]
MEAN = [0.408, 0.447, 0.470]
STD = [0.289, 0.274, 0.278]
IM_SHAPE = 512


def rescale(dets,meta,scale):


    dets = dets.detach().cpu().numpy()
    dets = dets.reshape(1,-1,dets.shape[2])
    dets = ctdet_post_process(
        dets.copy(), [meta['c']], [meta['s']],
        meta['out_height'], meta['out_width'], 80)

    for j in range(1, 81):
      dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
      dets[0][j][:, :4] /= scale
    return dets[0]


model = get_large_hourglass_net()
load_weights(PATH,model)

im = cv2.imread(IMG_PATH)
# # im = cv2.resize(im,(512,512))
# im = np.transpose(im,(2,0,1))
# im = np.expand_dims(im,axis=0)
# im = torch.Tensor(im)
# output = model(im)[-1]
# hm = output['hm'].sigmoid_()
# wh = output['wh']
# reg = output['reg']
detections = []
for scale in SCALES:
    images,meta = process(im,scale)
    print(images.shape)
    output = model(images)[-1]
    hm = output['hm'].sigmoid_()
    wh = output['wh']
    reg = output['reg']
    dets = ctdet_decode(hm, wh, reg=reg, K=100)
    dets = rescale(dets,meta,scale)
    detections.append(dets)


results = {}
for j in range(1, 80 + 1):
  results[j] = np.concatenate(
    [detection[j] for detection in detections], axis=0).astype(np.float32)
  # if len(SCALES) > 1:
  soft_nms(results[j], Nt=0.5, method=2)
scores = np.hstack(
  [results[j][:, 4] for j in range(1, 80 + 1)])
if len(scores) > 100:
  kth = len(scores) - 100
  thresh = np.partition(scores, kth)[kth]
  for j in range(1, 80 + 1):
    keep_inds = (results[j][:, 4] >= thresh)
    results[j] = results[j][keep_inds]
for i in results:
    boxes = results[i]
    if len(boxes) == 0:
        continue
    for box in boxes:
        x1,y1,x2,y2,score = int(box[0]),int(box[1]),int(box[2]),int(box[3]),float(box[4])
        if score < 0.5:
            continue

        cv2.rectangle(im, (x1,y2), (x2,y1), (255, 0, 0), 2)

cv2.imwrite('./test.jpg',im)
# cv2.imshow('1',im)
# cv2.waitKey(0)










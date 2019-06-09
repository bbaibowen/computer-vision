#gen anchors
import torch
import torch.nn as nn
import numpy as np
import itertools
from utils import box_encode

FEATURE_MAPS = [38, 19, 10, 5, 3, 1]
IMG_SIZE = 300
MIN_AREA = [30, 60, 111, 162, 213, 264]
MAX_AREA = [60, 111, 162, 213, 264, 315]
RATIOS = [[2,3], [2, 3], [2, 3], [2, 3], [2], [2]]
SCALES = [0.1,0.2]
STEP = [8, 16, 32, 64, 100, 300]
NUM_ANCHOES = len(RATIOS)


#和ssd一样，只不过都是生成6个
def gen_anchors():

    all_anchors = []
    for k,f in enumerate(FEATURE_MAPS):
        for i,j in itertools.product(range(f),range(f)):
            s = IMG_SIZE / STEP[k]
            x = (j + 0.5) / s
            y = (i + 0.5) / s

            first = MIN_AREA[k] / IMG_SIZE
            all_anchors.append([x,y,first,first])

            second = np.sqrt(first * (MAX_AREA[k] / IMG_SIZE))
            all_anchors.append([x,y,second,second])

            for v in RATIOS[k]:
                all_anchors.append([x,y,first * np.sqrt(v),first / np.sqrt(v)])
                all_anchors.append([x,y,first / np.sqrt(v),first * np.sqrt(v)])

    all_anchors = torch.Tensor(all_anchors)
    all_anchors = all_anchors.reshape((-1,4))
    return all_anchors




if __name__ == '__main__':
    all_anchors = gen_anchors()
    print(all_anchors,all_anchors.shape)
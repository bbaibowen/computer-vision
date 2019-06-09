import torch
import torch.utils.data as data
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import random
import math
import os
IMG_SHAPE = 300
VOC_CLASSES = ( '__background__', # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
_means = (104, 117, 123)
scale = 0.6
ROOT = './VOCdevkit/'
SETS = [('2007', 'trainval')]
class preproc:

    def __init__(self):
        self.means = _means
        self.resize = 300
        self.p = scale

    def __call__(self, image, targets):
        boxes = targets[:,:-1].copy()
        labels = targets[:,-1].copy()
        if len(boxes) == 0:
            targets = np.zeros((1,5))
            image = self.preproc_for_test(image, self.resize, self.means)
            return torch.from_numpy(image), targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:,:-1]
        labels_o = targets_o[:,-1]
        boxes_o[:, 0::2] /= width_o
        boxes_o[:, 1::2] /= height_o
        labels_o = np.expand_dims(labels_o,1)
        targets_o = np.hstack((boxes_o,labels_o))

        image_t, boxes, labels = self._crop(image, boxes, labels)
        image_t = self._distort(image_t)
        image_t, boxes = self._expand(image_t, boxes, self.means, self.p)
        image_t, boxes = self._mirror(image_t, boxes)
        #image_t, boxes = _mirror(image, boxes)

        height, width, _ = image_t.shape
        image_t = self.preproc_for_test(image_t, self.resize, self.means)
        boxes = boxes.copy()
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height
        b_w = (boxes[:, 2] - boxes[:, 0])*1.
        b_h = (boxes[:, 3] - boxes[:, 1])*1.
        mask_b= np.minimum(b_w, b_h) > 0.01
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b].copy()

        if len(boxes_t)==0:
            image = self.preproc_for_test(image_o, self.resize, self.means)
            return torch.from_numpy(image),targets_o

        labels_t = np.expand_dims(labels_t,1)
        targets_t = np.hstack((boxes_t,labels_t))

        return torch.from_numpy(image_t), targets_t

    def preproc_for_test(self,image, insize, mean):
        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[random.randrange(5)]
        image = cv2.resize(image, (insize, insize), interpolation=interp_method)
        image = image.astype(np.float32)
        image -= mean
        return image.transpose(2, 0, 1)

    def _crop(self,image, boxes, labels):
        height, width, _ = image.shape

        if len(boxes) == 0:
            return image, boxes, labels

        while True:
            mode = random.choice((
                None,
                (0.1, None),
                (0.3, None),
                (0.5, None),
                (0.7, None),
                (0.9, None),
                (None, None),
            ))

            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            for _ in range(50):
                scale = random.uniform(0.3, 1.)
                min_ratio = max(0.5, scale * scale)
                max_ratio = min(2, 1. / scale / scale)
                ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
                w = int(scale * ratio * width)
                h = int((scale / ratio) * height)

                l = random.randrange(width - w)
                t = random.randrange(height - h)
                roi = np.array((l, t, l + w, t + h))

                iou = self.matrix_iou(boxes, roi[np.newaxis])

                if not (min_iou <= iou.min() and iou.max() <= max_iou):
                    continue

                image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

                centers = (boxes[:, :2] + boxes[:, 2:]) / 2
                mask = np.logical_and(roi[:2] < centers, centers < roi[2:]) \
                    .all(axis=1)
                boxes_t = boxes[mask].copy()
                labels_t = labels[mask].copy()
                if len(boxes_t) == 0:
                    continue

                boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
                boxes_t[:, :2] -= roi[:2]
                boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
                boxes_t[:, 2:] -= roi[:2]

                return image_t, boxes_t, labels_t

    def matrix_iou(self,a,b):

        lt = np.maximum(a[:, np.newaxis, :2], b[:, :2]) # top-left
        rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:]) # right-bottom
        area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
        area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
        area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
        return area_i / (area_a[:, np.newaxis] + area_b - area_i)

    def _distort(self,image):
        def _convert(image, alpha=1, beta=0):
            tmp = image.astype(float) * alpha + beta
            tmp[tmp < 0] = 0
            tmp[tmp > 255] = 255
            image[:] = tmp

        image = image.copy()

        if random.randrange(2):
            _convert(image, beta=random.uniform(-32, 32))

        if random.randrange(2):
            _convert(image, alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if random.randrange(2):
            tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
            tmp %= 180
            image[:, :, 0] = tmp

        if random.randrange(2):
            _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        return image

    def _expand(self,image, boxes, fill, p):
        if random.random() > p:
            return image, boxes

        height, width, depth = image.shape
        for _ in range(50):
            scale = random.uniform(1, 4)

            min_ratio = max(0.5, 1. / scale / scale)
            max_ratio = min(2, scale * scale)
            ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
            ws = scale * ratio
            hs = scale / ratio
            if ws < 1 or hs < 1:
                continue
            w = int(ws * width)
            h = int(hs * height)

            left = random.randint(0, w - width)
            top = random.randint(0, h - height)

            boxes_t = boxes.copy()
            boxes_t[:, :2] += (left, top)
            boxes_t[:, 2:] += (left, top)

            expand_image = np.empty(
                (h, w, depth),
                dtype=image.dtype)
            expand_image[:, :] = fill
            expand_image[top:top + height, left:left + width] = image
            image = expand_image

            return image, boxes_t

    def _mirror(self,image, boxes):
        _, width, _ = image.shape
        if random.randrange(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes


class AnnotationTransform:


    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):

        res = np.empty((0,5))
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                #cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res,bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]

prepro = preproc()
target_trans = AnnotationTransform()

class VOC_load(data.Dataset):

    def __init__(self):
        self.root = ROOT
        self.prepro = prepro
        self.target_trans = target_trans
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.id = []
        for (y,n) in SETS:
            self.year = y
            rootpath = os.path.join(self.root, 'VOC' + y)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', 'train.txt')):
                self.id.append((rootpath, line.strip()))

    def __len__(self):
        return len(self.id)

    # def read_img(self,i):
    #
    #     return cv2.imread(self._imgpath % self.id[i])
    #
    # def read_anno(self,i):
    #     ann = ET.parse(self._annopath % self.id[i]).getroot()
    #     gt = self.target_trans(ann,1,1)
    #     return self.id[i][1],gt

    def __getitem__(self, item):
        img_id = self.id[item]
        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        h,w,_ = img.shape
        target = self.target_trans(target)
        img,target = self.prepro(img,target)

        return img,target



if __name__ == '__main__':
    test = VOC_load()
    img, target = test.__getitem__(5)
    print(img.shape,target.shape)
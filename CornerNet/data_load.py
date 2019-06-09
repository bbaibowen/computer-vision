import torch
import torch.utils.data as data
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image,ImageDraw
import random
import math
import torchvision.transforms as transforms
import os
IMG_SHAPE = 300
VOC_CLASSES = ( '__background__', # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
ROOT = '../des/VOCdevkit/'
SETS = [('2007', 'trainval')]
transform = transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
                            ])

def resize(img, boxes, size, max_size=1000):
    '''Resize the input PIL image to the given size.

    Args:
      img: (PIL.Image) image to be resized.
      boxes: (tensor) object boxes, sized [#ojb,4].
      size: (tuple or int)
        - if is tuple, resize image to the size.
        - if is int, resize the shorter side to the size while maintaining the aspect ratio.
      max_size: (int) when size is int, limit the image longer size to max_size.
                This is essential to limit the usage of GPU memory.
    Returns:
      img: (PIL.Image) resized image.
      boxes: (tensor) resized boxes.
    '''
    w, h = img.size
    if isinstance(size, int):
        size_min = min(w,h)
        size_max = max(w,h)
        sw = sh = float(size) / size_min
        if sw * size_max > max_size:
            sw = sh = float(max_size) / size_max
        ow = int(w * sw + 0.5)
        oh = int(h * sh + 0.5)
    else:
        ow, oh = size
        sw = float(ow) / w
        sh = float(oh) / h
    return img.resize((ow,oh), Image.BILINEAR), \
           boxes*torch.Tensor([sw,sh,sw,sh])


def random_flip(img, boxes):
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        w = img.width
        xmin = w - boxes[:,2]
        xmax = w - boxes[:,0]
        boxes[:,0] = xmin
        boxes[:,2] = xmax
    return img, boxes

def random_crop(img, boxes):
    '''Crop the given PIL image to a random size and aspect ratio.
    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made.
    Args:
      img: (PIL.Image) image to be cropped.
      boxes: (tensor) object boxes, sized [#ojb,4].
    Returns:
      img: (PIL.Image) randomly cropped image.
      boxes: (tensor) randomly cropped boxes.
    '''
    success = False
    for attempt in range(10):
        area = img.size[0] * img.size[1]
        target_area = random.uniform(0.56, 1.0) * area
        aspect_ratio = random.uniform(3. / 4, 4. / 3)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5:
            w, h = h, w

        if w <= img.size[0] and h <= img.size[1]:
            x = random.randint(0, img.size[0] - w)
            y = random.randint(0, img.size[1] - h)
            success = True
            break

    # Fallback
    if not success:
        w = h = min(img.size[0], img.size[1])
        x = (img.size[0] - w) // 2
        y = (img.size[1] - h) // 2

    img = img.crop((x, y, x+w, y+h))
    boxes -= torch.Tensor([x,y,x,y])
    boxes[:,0::2].clamp_(min=0, max=w-1)
    boxes[:,1::2].clamp_(min=0, max=h-1)
    return img, boxes


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

        return torch.Tensor(res)

class VOC_load(data.Dataset):

    def __init__(self):
        self.root = ROOT
        self.target_trans = AnnotationTransform()
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

    def __getitem__(self, item):
        img_id = self.id[item]
        target = ET.parse(self._annopath % img_id).getroot()
        img = Image.open(self._imgpath % img_id)
        boxes = self.target_trans(target)


        img,boxes[:,:4] = random_flip(img,boxes[:,:4])
        img,boxes[:,:4] = random_crop(img,boxes[:,:4])
        img,boxes[:,:4] = resize(img,boxes[:,:4],(511,511))
        labels = boxes[:,-1]
        boxes = boxes[:,:4]
        img = transform(img)

        return img,boxes,labels


def collate_fn(batch):

    max_tag = 100
    scale = 128 / 511
    imgs = []
    boxes = []
    labels = []
    for i in batch:
        imgs += [i[0]]
        boxes += [i[1]]
        labels += [i[2]]
    N = len(imgs)
    im_temp = torch.zeros(N,3,511,511)
    tl_heats = torch.zeros((N,80,128,128))
    br_heats = torch.zeros((N,80,128,128))

    tl_regrs = torch.zeros((N, max_tag, 2))
    br_regrs = torch.zeros((N, max_tag, 2))

    tl_tags = torch.zeros((N, max_tag))
    br_tags = torch.zeros((N, max_tag))

    tag_masks = torch.zeros((N, max_tag))
    tag_lens = torch.zeros((N,))
    for i in range(N):
        im_temp[i] = imgs[i]
        cur_labels = labels[i]
        for j in range(len(boxes[i])):
            label = cur_labels[j].item()
            xtl, ytl = boxes[i][j][0], boxes[i][j][1]
            xbr, ybr = boxes[i][j][2], boxes[i][j][3]
            fxtl = (xtl * scale)
            fytl = (ytl * scale)
            fxbr = (xbr * scale)
            fybr = (ybr * scale)
            xtl = int(fxtl)
            ytl = int(fytl)
            xbr = int(fxbr)
            ybr = int(fybr)

            #gaussian?
            tl_heats[i,int(label),xtl,ytl] = 1
            br_heats[i,int(label),xbr,ybr] = 1

            tag_id = tag_lens[i].long().item()
            tl_regrs[i,tag_id,:] = torch.Tensor([fxtl - xtl,fytl - ytl])
            br_regrs[i,tag_id,:] = torch.Tensor([fxbr - xbr,fybr - ybr])
            tl_tags[i,tag_id] = ytl * 128 + xtl
            br_tags[i,tag_id] = ybr * 128 + xbr

            tag_lens[i] += 1
    for i in range(N):
        tag_len = tag_lens[i].long().item()
        tag_masks[i,:tag_len] = 1

    return im_temp,[tl_heats, br_heats, tl_tags, br_tags, tl_regrs, br_regrs, tag_masks]


def vis(img,boxes):
    draw = ImageDraw.Draw(img)
    for i in boxes:
        draw.rectangle((int(i[0]), int(i[1]), int(i[2]), int(i[3])), outline='red')
    Image._show(img)


if __name__ == '__main__':
    # datasets = VOC_load()
    data_io = VOC_load()
    trainloader = data.DataLoader(data_io, batch_size=1, shuffle=True,
                                              collate_fn=collate_fn)
    for idx, ip_dict in enumerate(trainloader):

        im = ip_dict[0]
        targets = ip_dict[1]
        print(targets)


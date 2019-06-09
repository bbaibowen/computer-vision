import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os
# import PIL.Image as P


CLASSES = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
IMAGE_SHAPE = (600,800)



DATA_PATH = 'C:\\Users\\ZD\\Desktop\\FasterRCNN_TF-master\\FasterRCNN_TF-master\\experiments\\experiment1\\data'
XML_PATH = DATA_PATH + '\\xmls\\'
IMAGE_PATH = DATA_PATH + '\\images\\'

def load_img(path):

    img = cv2.imread(IMAGE_PATH + path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img, np.float32)
    img -= means
    img = cv2.resize(img, (IMAGE_SHAPE[1], IMAGE_SHAPE[0]))
    img = np.expand_dims(img,axis=0)
    return img

def load_xml(path):
    tree = ET.parse(path)
    size = tree.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    bili = (IMAGE_SHAPE[0] / h) if (h / w) > (IMAGE_SHAPE[0] / IMAGE_SHAPE[1]) else (IMAGE_SHAPE[1] / w)
    objs = tree.findall('object')
    boxes = np.zeros((len(objs),5),dtype = np.float32)
    for id ,obj in enumerate(objs):
        bbox = obj.find('bndbox')
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        cls = CLASSES.index(obj.find('name').text.strip())

        boxes[id,:] = [x1 * bili,y1 * bili,x2 * bili,y2 * bili,cls]

    return boxes

means = np.array((123., 117., 104.))

def get_data(name):

    xml_name = name.split('.')[0].strip() + '.xml'
    boxes = load_xml(XML_PATH + xml_name)
    img = load_img(name)

    return boxes,img



if __name__ == '__main__':
    names = os.listdir(IMAGE_PATH)
    # print(names)
    for name in names:

        b,i = get(name)
        print(b.shape,i.shape)
        print('success')
import os
import h5py
import numpy as np
import xml.etree.ElementTree as ElementTree
import matplotlib.pyplot as plt


CLASS = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
voc_path = '../../VOC2012'

def get_boxes_for_id(voc_path,id):

    fname = voc_path + '/Annotations/{}.xml'.format(id)
    with open(fname) as in_file:
        xml_tree = ElementTree.parse(in_file)

    root = xml_tree.getroot()
    boxes = []
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        label = obj.find('name').text
        if label not in CLASS or int(difficult) == 1:  # exclude difficult or unlisted classes
            continue
        xml_box = obj.find('bndbox')
        bbox = (CLASS.index(label), int(xml_box.find('xmin').text),
                int(xml_box.find('ymin').text), int(xml_box.find('xmax').text),
                int(xml_box.find('ymax').text))
        boxes.extend(bbox)
    return np.array(boxes)

def get_image_for_id(voc_path,id):
    fname = voc_path + '/JPEGImages/{}.jpg'.format(id)
    with open(fname, 'rb') as in_file:
        data = in_file.read()
    return np.fromstring(data, dtype='uint8')


#获取txt文件内图片名称
def get_ids(voc_path,datasets):
    ids = []
    # for image_set in datasets:
    id_file = voc_path + '/ImageSets/Main/{}.txt'.format(datasets)
    with open(id_file, 'r') as image_ids:
        ids.extend(map(str.strip, image_ids.readlines()))
    return ids

def add_to_dataset(voc_path, ids, images, boxes, start=0):
    """Process all given ids and adds them to given datasets."""
    for i, voc_id in enumerate(ids):
        image_data = get_image_for_id(voc_path, voc_id)
        image_boxes = get_boxes_for_id(voc_path, voc_id)
        images[start + i] = image_data
        boxes[start + i] = image_boxes

def main(voc_path):
    train_ids = get_ids(voc_path,'train')
    fname = 'voc2007.hdf5'
    voc_h5file = h5py.File(fname,'w')
    uint8_dt = h5py.special_dtype(vlen = np.dtype('uint8'))
    vlen_int_dt = h5py.special_dtype(
        vlen=np.dtype(int))
    train_group = voc_h5file.create_group('train')
    voc_h5file.attrs['classes'] = np.string_(str.join(',', CLASS))

    train_img = train_group.create_dataset(
        'images',shape=(len(train_ids),),dtype=uint8_dt
    )
    train_boxes = train_group.create_dataset(
        'boxes',shape=(len(train_ids),),dtype=vlen_int_dt
    )
    add_to_dataset(voc_path,train_ids,train_img,train_boxes)
    voc_h5file.close()
    print('done')

if __name__ == '__main__':
    main(voc_path)  #保存为h5

import numpy as np
import os
import cv2
import xml.etree.ElementTree as ET
import copy

class load_data(object):
    def __init__(self,path,batch,CLASS):

        self.devkil_path = path + '/VOCdevkit'
        self.data_path = self.devkil_path + '/VOC2007'
        self.img_size = 448
        self.batch = batch
        self.CLASS = CLASSES
        self.n_class = len(CLASS)
        self.class_id = dict(zip(CLASS, range(self.n_class)))
        self.id = 0
        self.run_this()


    def load_img(self,PATH):
        im = cv2.imread(PATH)
        im = cv2.resize(im,(self.img_size,self.img_size))
        im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        im = np.multiply(1./255.,im)
        return im


    def load_xml(self,index):
        path = self.data_path + '/JPEGImages/' + index  + '.jpg'
        xml_path = self.data_path + '/Annotations/' + index + '.xml'
        img = cv2.imread(path)
        w = self.img_size / img.shape[0]
        h = self.img_size / img.shape[1]

        label = np.zeros((7,7,25))
        tree = ET.parse(xml_path)
        objs = tree.findall('object')
        for i in objs:
            box = i.find('bndbox')
            x1 = max(min((float(box.find('xmin').text) - 1) * w,self.img_size-1),0)
            y1 = max(min((float(box.find('ymin').text) - 1) * h,self.img_size-1),0)
            x2 = max(min((float(box.find('xmax').text) - 1) * w,self.img_size-1),0)
            y2 = max(min((float(box.find('ymax').text) - 1) * w,self.img_size-1),0)
            boxes = [(x1+x2)/2.,(y1+y2)/2.,x2-x1,y2-y1]
            cls_id = self.class_id[i.find('name').text.lower().strip()]

            x_id = int(boxes[0] * 7 / self.img_size)
            y_id = int(boxes[1] * 7 / self.img_size)

            if label[y_id,x_id,0] == 1:
                continue
            label[y_id,x_id,0] = 1  #con
            label[y_id,x_id,1:5] = boxes
            label[y_id,x_id,5:cls_id] = 1

        return label,len(objs)


    def load_label(self):
        path = self.data_path + '/ImageSets/Main/trainval.txt'

        with open(path,'r') as f:
            index = [x.strip() for x in f.readlines()]
        labels = []
        for i in index:
            la,num = self.load_xml(i)
            if num == 0:
                continue
            img_name = self.data_path + '/JPEGImages/' + i +'.jpg'
            labels.append({'img_name':img_name,
                           'label':la})
        return labels

    def run_this(self):
        labels = self.load_label()
        np.random.shuffle(labels)
        self.truth_label = labels
        return labels


    #取数据
    def get_data(self):
        img = np.zeros((self.batch,self.img_size,self.img_size,3))
        labels = np.zeros((self.batch,7,7,25))
        times = 0
        while times < self.batch:
            img_name = self.truth_label[self.id]['img_name']
            img[times,:,:,:] = self.load_img(img_name)
            labels[times,:,:,:] = self.truth_label[self.id]['label']
            times += 1
            self.id += 1

            # if self.id > len(self.truth_label):
            #     np.random.shuffle(self.truth_label)
        return img,labels



if __name__ == '__main__':
    img = cv2.imread('../test/cat.jpg')
    cv2.imshow('1',img)
    img = img[:,::-1,:]

    cv2.imshow('im',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']
    data_path = '../data/pascal_voc'
    test = load_data(data_path,10,CLASSES)
    img, labels = test.get_data()
    print(img.shape,labels.shape)





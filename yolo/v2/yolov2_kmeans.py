import numpy as np
import xml.etree.ElementTree as ET
import glob
import random

def cas_iou(box,cluster):
    x = np.minimum(cluster[:,0],box[0])
    y = np.minimum(cluster[:,1],box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:,0] * cluster[:,1]
    iou = intersection / (area1 + area2 -intersection)

    return iou

def avg_iou(box,cluster):
    return np.mean([np.max(cas_iou(box[i],cluster)) for i in range(box.shape[0])])


def kmeans(box,k):
    row = box.shape[0]
    distance = np.empty((row,k))
    last_clu = np.zeros((row,))

    np.random.seed()

    cluster = box[np.random.choice(row,k,replace = False)]
    # cluster = random.sample(row, k)
    while True:
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i],cluster)

        near = np.argmin(distance,axis=1)

        if (last_clu == near).all():
            break

        for j in range(k):
            cluster[j] = np.median(
                box[near == j],axis=0)

        last_clu = near

    return cluster



def load_data(path):
    data = []
    for xml_file in glob.glob('{}/*xml'.format(path)):
        tree = ET.parse(xml_file)
        height = int(tree.findtext('./size/height'))
        width = int(tree.findtext('./size/width'))
        for obj in tree.iter('object'):
            xmin = int(obj.findtext('bndbox/xmin')) / width
            ymin = int(obj.findtext('bndbox/ymin')) / height
            xmax = int(obj.findtext('bndbox/xmax')) / width
            ymax = int(obj.findtext('bndbox/ymax')) / height

            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)

            data.append([xmax-xmin,ymax-ymin])
    return np.array(data)


if __name__ == '__main__':
    anchors_num = 5
    path = '../../VOC2012/Annotations'
    data = load_data(path)
    # print(data)
    out = kmeans(data,anchors_num)
    print('acc:{:.2f}%'.format(avg_iou(data,out) * 100))
    print(out)
    print('box',out[:,0] * 13,out[:,1] * 13)
    ratios = np.around(out[:,0]/out[:,1],decimals=2).tolist()
    print('ratios:',sorted(ratios))








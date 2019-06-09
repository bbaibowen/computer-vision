import tensorflow as tf
import numpy as np
from yolo_v2 import yolov2
from load_data import load_data
from keras import backend as K
import PIL.Image as P
import h5py
import io
from keras.layers import Input,Lambda
from keras.models import Model


YOLO_ANCHORS = np.array(
    ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
     (7.88282, 3.52778), (9.77052, 9.16828)))


CLASS_PATH = 'pascal_classes.txt'
h5_path = 'voc2007.hdf5'
with open(CLASS_PATH) as f:
    class_names = f.readlines()
    f.close()
class_names = [c.strip() for c in class_names]
MAX_STEP = 100000
BATCH = 10
sess = tf.Session()

#加载数据
load_io = load_data(1,h5_path)
images,boxes = load_io.get()
bos = np.expand_dims(boxes,0)
op = tf.train.GradientDescentOptimizer(1e-3)

'''
预训练：
    1、在imagenet数据集用darknet-19训练160epo ，input 224,224,3  learning_rate:1e-1
    2、继续在imagenet数据集上训练10epo input:448,448,3  
    3、darknet-19从分类模型转到检测模型
    
匹配原则：
    当图片的中心落在一个cell,那么这个cell的5 anchors，iou最大的那个先验框的边界框就会和gt进匹配  正样本
    在剩下anchors，IOU < 0.6 --->这个边界框就被标记为背景
    在剩下的IOU > 0.6   ----->IOU值的LOSS计算    负样本
'''




yolo = yolov2('voc')
net,x = yolo.darknet()
init = tf.global_variables_initializer()
# detectors_mask_shape = (13, 13, 5, 1)
# matching_boxes_shape = (13, 13, 5, 5)

# true_box = tf.placeholder(tf.float32,[None,5])
# detectors_mask, matching_true_boxes = yolo.preprocess_true_boxes(boxes,YOLO_ANCHORS)
# print(detectors_mask.shape,matching_true_boxes.shape)
# print(sess.run(K.shape(boxes)))

# loss = yolo.loss(net,bos,detectors_mask,matching_true_boxes,YOLO_ANCHORS,yolo.num_class)


# op = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)
# op = tf.train.GradientDescentOptimizer(1e-3)

# sess.run(init)
#
# loss_val = sess.run(loss,feed_dict={x:images})
# print(loss_val)


def main(sess,op,init,net,x):
    sess.run(init)
    for i in range(MAX_STEP):
        images, boxes = load_io.get()
        bos = np.expand_dims(boxes, 0)
        detectors_mask, matching_true_boxes = yolo.preprocess_true_boxes(boxes, YOLO_ANCHORS)
        loss = yolo.loss(net, bos, detectors_mask, matching_true_boxes, YOLO_ANCHORS, yolo.num_class)
        op.minimize(loss)
        # sess.run(init)
        loss_val = sess.run(loss, feed_dict={x: images})
        print(loss_val)

if __name__ == '__main__':
    main(sess,op,init,net,x)





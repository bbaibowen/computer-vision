import tensorflow as tf
import numpy as np
import cv2
import colorsys
import random
import keras



slim = tf.contrib.slim

'''
darknet-53:深度残差网络 + 多尺度特征检测 + FPN上采样
            一般来说，网络越深，检测到的特征越细 ----> 越好  
            alexnet ----  vgg   --->
        15层的网络   ---->    12个层这个地方就出现一个饱和现象，13、14、15：网络退化的现象
        跳跃结构 ：  h(x) = f(x) + x
                    h(x) - x  ----->   0
                    
            现在吃得很饱了，我妈妈又给做个十几个鸡腿 ---->  可能出现吐的情况，可能把之前吃饱也吐出来  午餐
            跳跃结构：现在吃得很饱了，在下一顿（晚餐）给我做了十几个鸡腿 胃容量=0，吃十几个鸡腿，肯定吃得很香的啊！！！
             -------》  吃饱了       
FPN网络：多尺度特征融合网络


yolov3: 深度残差网络 + 多尺度特征融合
maxpooling :过滤好处    信息丢失

'''




class Yolo(object):

    def __init__(self,data_name):
        if data_name == 'coco':
            self.CLASS = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train',
                          'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
                          'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                          'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                          'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                          'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                          'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                          'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                          'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant',
                          'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse',
                          'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                          'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
                          'hair drier', 'toothbrush']
            self.num_class = len(self.CLASS)
        elif data_name == 'voc':
            self.CLASS = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                          "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                          "tvmonitor"]
            self.num_class = len(self.CLASS)
        elif data_name != 'voc' and data_name != 'coco':
            raise ValueError("Input must be 'coco' or 'voc'")
        self.obj_threshold = 0.6
        self.nms_threshold = 0.5
        self.batch_norm_decay = 0.9
        self.batch_norm_epsilon = 1e-05
        self.alpha = 0.1
        self.batch_norm = {
            'decay':self.batch_norm_decay,
            'epsilon':self.batch_norm_epsilon,
            'scale':True,
            'is_training':False,
            'fused':None
        }
        self._ANCHORS = [(10, 13), (16, 30), (33, 23),
                        (30, 61), (62, 45), (59, 119),
                        (116, 90), (156, 198), (373, 326)]

        self.anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

        self.feature_map = [13,26,52]
        self.img_size = (416,416)
        self.stride = (32,16,8)  #下采样倍数


    def conv2d(self,x,num_filter,k_size,s):

        return slim.conv2d((x if s == 1 or s == 0 else self.padding(x,k_size)),
                           num_filter,
                           k_size,
                           stride=s,
                           padding = ('SAME' if s == 1 else 'VALID'))

    @tf.contrib.framework.add_arg_scope
    def padding(self,x,k_size):
        pad_t = k_size - 1
        pad1 = pad_t // 2
        pad2 = pad_t - pad1
        return tf.pad(x,[[0,0],[pad1,pad2],[pad1,pad2],[0,0]])

    def net_block(self,x,num_filter):
        shortcut = x
        net = self.conv2d(x,num_filter,1,1)
        net = self.conv2d(net,num_filter * 2,3,1)

        return shortcut + net

    def features(self,net,num_filter):
        net = self.conv2d(net,num_filter,1,1)
        net = self.conv2d(net,num_filter * 2,3,1)
        net = self.conv2d(net,num_filter,1,1)
        net = self.conv2d(net,num_filter * 2,3,1)
        net = self.conv2d(net,num_filter,1,1)

        get1 = net

        net = self.conv2d(net,num_filter * 2,3,1)
        get2 = slim.conv2d(net, 3 * (5 + self.num_class), 1, stride=1, normalizer_fn=None,
                            activation_fn=None, biases_initializer=tf.zeros_initializer())

        return get1,get2


    def darknet_53(self,net):

        net = self.conv2d(net,32,3,1)
        net = self.conv2d(net,64,3,2)
        net = self.net_block(net,32)  #1x
        net = self.conv2d(net,128,3,2)

        for i in range(2):  #2x
            net = self.net_block(net,64)

        net = self.conv2d(net,256,3,2)

        for i in range(8):  #8x
            net = self.net_block(net,128)

        feats_1 = net

        net = self.conv2d(net,512,3,2)

        for i in range(8):  #8x
            net = self.net_block(net,256)

        feats_2 = net

        net = self.conv2d(net,1024,3,2)

        for i in range(4): #4x
            net = self.net_block(net,512)

        return feats_1,feats_2,net

    # 最近邻插值上采样
    def _upsample(self,net, out_shape):
        net = tf.image.resize_nearest_neighbor(net, (out_shape[1], out_shape[2]))
        net = tf.identity(net, name='upsampled')
        return net

    def main_net(self,x):
        with slim.arg_scope([slim.conv2d, slim.batch_norm, self.padding]):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=self.batch_norm,
                            biases_initializer=None, activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=self.alpha)):
                with tf.variable_scope('detector'):
                    with tf.variable_scope('darknet-53'):
                        feats_1, feats_2, net = self.darknet_53(x)

                    with tf.variable_scope('yolo-v3'):
                        #第一个特征图  13, 13,
                        net, feature1 = self.features(net,512)

                        net = self.conv2d(net,256,1,1)
                        net = self._upsample(net,feats_2.get_shape().as_list())
                        net = tf.concat([net,feats_2],axis=-1)

                        #第二个特征图  26, 26, )
                        net,feature2 = self.features(net,256)

                        net = self.conv2d(net,128,1,1)
                        net = self._upsample(net,feats_1.get_shape().as_list())
                        net = tf.concat([net,feats_1],axis=-1)

                        #第三个特征图  52, 52,
                        net,feature3 = self.features(net,128)


                        return feature1,feature2,feature3

    #################
    def decode_layer(self,scale_num,scale):
        anchors = np.array(self._ANCHORS).reshape(-1,2)[self.anchor_mask[scale_num]]
        len_anchors = len(anchors)


        anchors = [(i[0] / self.stride[scale_num],i[1] / self.stride[scale_num]) for i in anchors]

        #(-1,3*num_anchor,85)
        pred = tf.reshape(scale,[-1,len_anchors * self.feature_map[scale_num] * self.feature_map[scale_num],5 + self.num_class])
        print('pred',pred)
        box_centers, box_sizes, confidence, classes = tf.split(pred, [2, 2, 1, self.num_class], axis=-1)

        grid_x = tf.range(self.feature_map[scale_num],dtype=tf.float32)
        grid_y = tf.range(self.feature_map[scale_num],dtype=tf.float32)

        x,y = tf.meshgrid(grid_x,grid_y)

        x_offset = tf.reshape(x,(-1,1))
        y_offset = tf.reshape(y,(-1,1))

        offset = tf.concat([x_offset,y_offset],axis=-1)
        offset = tf.reshape(tf.tile(offset,[1,len_anchors]),[1,-1,2])

        box_centers = tf.nn.sigmoid(box_centers)
        box_centers += offset

        #映射回原图
        box_centers *= ([self.stride[scale_num]] * 2)


        anchors = tf.tile(anchors,[self.feature_map[scale_num] * self.feature_map[scale_num],1])

        box_sizes = tf.exp(box_sizes) * anchors

        box_sizes *= self.stride[scale_num]

        confidence = tf.nn.sigmoid(confidence)

        classes = tf.nn.sigmoid(classes)

        pre = tf.concat([box_centers,box_sizes,confidence,classes],axis=-1)

        return pre  #3x13x13+26x26x3+52x52x3


    def decode(self,scale):
        pred = []

        for i in range(3):
            pre = self.decode_layer(i,scale[i])

            pred.append(pre)

        pred = tf.concat(pred,axis=1)

        return pred

    def box_change(self,message):

        x, y, w, h, c = tf.split(message, [1, 1, 1, 1, -1], axis=-1)
        xmin = x - w * .5
        ymin = y - h * .5
        xmax = x + w * .5
        ymax = y + h * .5

        pred = tf.concat(
            [tf.concat([xmin,ymin,xmax,ymax],axis=-1),c]
        ,axis=-1)

        return pred

    def Nms(self,pred):

        boxes,confidence,cls_prob = tf.split(pred,[4,1,-1],axis=-1)

        score = confidence * cls_prob

        mask = score > self.obj_threshold

        all_cls_index = []
        all_scores = []
        all_boxes = []

        for i in range(len(self.CLASS)):
            cls_score = tf.boolean_mask(score[...,i],mask[...,i])
            cls_box = tf.boolean_mask(boxes,mask[...,i])

            nms = tf.image.non_max_suppression(cls_box,cls_score,
                                               max_output_size=30,
                                               iou_threshold=self.nms_threshold)
            sc = tf.gather(cls_score,nms)
            box = tf.gather(cls_box,nms)
            index = tf.ones_like(cls_score) * i

            all_cls_index.append(index)
            all_boxes.append(box)
            all_scores.append(sc)

        return (tf.concat(all_boxes,axis=0),
                tf.concat(all_scores,axis=0),
                tf.cast(tf.concat(all_cls_index,axis=0),tf.int32))

    def _draw_box(self, classes_index, boxes, scores, img, input_shape, classes):
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / float(len(classes)), 1., 1.)
                      for x in range(len(classes))]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.

        # h_ratio、w_ratio
        ratio = np.array(img.shape) / input_shape

        thick = int((img.shape[0] + img.shape[1]) / 700)
        for i in range(len(boxes)):
            x1 = int(boxes[i][0] * ratio[1])
            y1 = int(boxes[i][1] * ratio[0])
            x2 = int(boxes[i][2] * ratio[1])
            y2 = int(boxes[i][3] * ratio[0])

            cv2.rectangle(img, (x1, y1), (x2, y2), colors[classes_index[i].tolist()], thick)
            text = "%s : %f" % (classes[classes_index[i].tolist()], scores[i])
            cv2.putText(img, text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[classes_index[i].tolist()],
                        1)

            # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thick)
            # text = "%s : %f" % (classes[classes_index[i].tolist()], scores[i])
            # cv2.putText(img, text, (x1 + 5, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * img.shape[0], (255,0,0), 2)

        cv2.imshow('yolov3', img)
        cv2.waitKey(0)





if __name__ == '__main__':
    data_name = 'coco'
    ckpt_path = '../../model/yolo/v3/yolov3.ckpt'
    _img = cv2.imread('../../road.jpg')
    img = cv2.cvtColor(_img,cv2.cv2.COLOR_BGR2RGB).astype(np.float32)
    img = cv2.resize(img,(416,416))
    img = np.expand_dims(img,axis=0)
    img = img / 255.

    yolo = Yolo(data_name)
    x = tf.placeholder(dtype=tf.float32,shape=[None,416,416,3])

    feature1, feature2, feature3 = yolo.main_net(x)
    print(feature1, feature2, feature3)
    scale = [feature1,feature2,feature3]

    pred = yolo.decode(scale)
    print(pred)

    pred = yolo.box_change(pred)
    print(pred)

    box,scores,index = yolo.Nms(pred)


    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, ckpt_path)

    box, scores, index = sess.run([box,scores,index],feed_dict={x:img})

    yolo._draw_box(index,box,scores,_img,416,yolo.CLASS,)



    print('done')



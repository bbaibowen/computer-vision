import tensorflow as tf
import numpy as np
import cv2
from keras import backend as K
'''
运行这部分就能进行test，对于test而言，只有这个文件，
除了这个文件外其他文件都是train的组成部分
'''


def leaky_relu(x):    #leaky relu激活函数，leaky_relu激活函数一般用在比较深层次神经网络中
    return tf.maximum(0.1*x,x)

class yolov2(object):

    def __init__(self,cls_name):

        self.anchor_size = [[0.57273, 0.677385], #coco
                           [1.87446, 2.06253],
                           [3.33843, 5.47434],
                           [7.88282, 3.52778],
                           [9.77052, 9.16828]]
        self.num_anchors = len(self.anchor_size)
        if cls_name == 'coco':
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
                          'hair drier', 'toothbrush']  #coco
            self.f_num = 425

        else:
            self.CLASS = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
            self.f_num = 125

        self.num_class = len(self.CLASS)
        self.feature_map_size = (13,13)
        self.object_scale = 5. #'物体位于gird cell时计算置信度的修正系数'
        self.no_object_scale = 1.   #'物体位于gird cell时计算置信度的修正系数'
        self.class_scale = 1.  #'计算分类损失的修正系数'
        self.coordinates_scale = 1.  #'计算坐标损失的修正系数'


#################################NewWork

    def conv2d(self,x,filters_num,filters_size,pad_size=0,stride=1,batch_normalize=True,activation=leaky_relu,use_bias=False,name='conv2d'):

        if pad_size > 0:
            x = tf.pad(x,[[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])

        out = tf.layers.conv2d(x,filters=filters_num,kernel_size=filters_size,strides=stride,padding='VALID',activation=None,use_bias=use_bias,name=name)
        # BN应该在卷积层conv和激活函数activation之间,
        # (后面有BN层的conv就不用偏置bias，并激活函数activation在后)
        if batch_normalize:
            out = tf.layers.batch_normalization(out,axis=-1,momentum=0.9,training=False,name=name+'_bn')
        if activation:
            out = activation(out)
        return out

    def maxpool(self,x, size=2, stride=2, name='maxpool'):
        return tf.layers.max_pooling2d(x, pool_size=size, strides=stride,name=name)

    # passthrough
    def passthrough(self,x, stride):
        return tf.space_to_depth(x, block_size=stride)
        #或者tf.extract_image_patches

    def darknet(self):

        x = tf.placeholder(dtype=tf.float32,shape=[None,416,416,3])

        net = self.conv2d(x, filters_num=32, filters_size=3, pad_size=1,
                     name='conv1')
        net = self.maxpool(net, size=2, stride=2, name='pool1')

        net = self.conv2d(net, 64, 3, 1, name='conv2')
        net = self.maxpool(net, 2, 2, name='pool2')

        net = self.conv2d(net, 128, 3, 1, name='conv3_1')
        net = self.conv2d(net, 64, 1, 0, name='conv3_2')
        net = self.conv2d(net, 128, 3, 1, name='conv3_3')
        net = self.maxpool(net, 2, 2, name='pool3')

        net = self.conv2d(net, 256, 3, 1, name='conv4_1')
        net = self.conv2d(net, 128, 1, 0, name='conv4_2')
        net = self.conv2d(net, 256, 3, 1, name='conv4_3')
        net = self.maxpool(net, 2, 2, name='pool4')

        net = self.conv2d(net, 512, 3, 1, name='conv5_1')
        net = self.conv2d(net, 256, 1, 0, name='conv5_2')
        net = self.conv2d(net, 512, 3, 1, name='conv5_3')
        net = self.conv2d(net, 256, 1, 0, name='conv5_4')
        net = self.conv2d(net, 512, 3, 1, name='conv5_5')  #

        # 这一层特征图，要进行后面passthrough
        shortcut = net
        net = self.maxpool(net, 2, 2, name='pool5')  #

        net = self.conv2d(net, 1024, 3, 1, name='conv6_1')
        net = self.conv2d(net, 512, 1, 0, name='conv6_2')
        net = self.conv2d(net, 1024, 3, 1, name='conv6_3')
        net = self.conv2d(net, 512, 1, 0, name='conv6_4')
        net = self.conv2d(net, 1024, 3, 1, name='conv6_5')


        # 训练检测网络时去掉了分类网络的网络最后一个卷积层，
        # 在后面增加了三个卷积核尺寸为3 * 3，卷积核数量为1024的卷积层，并在这三个卷积层的最后一层后面跟一个卷积核尺寸为1 * 1
        # 的卷积层，卷积核数量是（B * （5 + C））。
        # 对于VOC数据集，卷积层输入图像尺寸为416 * 416
        # 时最终输出是13 * 13
        # 个栅格，每个栅格预测5种boxes大小，每个box包含5个坐标值和20个条件类别概率，所以输出维度是13 * 13 * 5 * （5 + 20）= 13 * 13 * 125。
        #
        # 检测网络加入了passthrough layer，从最后一个输出为26 * 26 * 512
        # 的卷积层连接到新加入的三个卷积核尺寸为3 * 3
        # 的卷积层的第二层，使模型有了细粒度特征。

        # 下面这部分主要是training for detection
        net = self.conv2d(net, 1024, 3, 1, name='conv7_1')
        net = self.conv2d(net, 1024, 3, 1, name='conv7_2')

        # shortcut增加了一个中间卷积层，先采用64个1*1卷积核进行卷积，然后再进行passthrough处理
        # 这样26*26*512 -> 26*26*64 -> 13*13*256的特征图
        shortcut = self.conv2d(shortcut, 64, 1, 0, name='conv_shortcut')
        shortcut = self.passthrough(shortcut, 2)

        # 连接之后，变成13*13*（1024+256）
        net = tf.concat([shortcut, net],
                        axis=-1)  # channel整合到一起，concatenated with the original features，passthrough层与ResNet网络的shortcut类似，以前面更高分辨率的特征图为输入，然后将其连接到后面的低分辨率特征图上，
        net = self.conv2d(net, 1024, 3, 1, name='conv8')

        # detection layer: 最后用一个1*1卷积去调整channel，该层没有BN层和激活函数，变成: S*S*(B*(5+C))，在这里为：13*13*425
        output = self.conv2d(net, filters_num=self.f_num, filters_size=1, batch_normalize=False, activation=None,
                        use_bias=True, name='conv_dec')

        return output,x




#生成anchor  --->  decode
    def decode(self,net):

        self.anchor_size = tf.constant(self.anchor_size,tf.float32)

        net = tf.reshape(net, [-1, 13 * 13, self.num_anchors, self.num_class + 5]) #[batch,169,5,85]

        # 偏移量、置信度、类别
        #中心坐标相对于该cell坐上角的偏移量，sigmoid函数归一化到(0,1)
        xy_offset = tf.nn.sigmoid(net[:, :, :, 0:2])
        wh_offset = tf.exp(net[:, :, :, 2:4])
        obj_probs = tf.nn.sigmoid(net[:, :, :, 4])  # 置信度,这个东西就是相当于v1中的confidence
        class_probs = tf.nn.softmax(net[:, :, :, 5:])  #

        # 在feature map对应坐标生成anchors，每个坐标五个
        height_index = tf.range(self.feature_map_size[0], dtype=tf.float32)
        width_index = tf.range(self.feature_map_size[1], dtype=tf.float32)

        x_cell, y_cell = tf.meshgrid(height_index, width_index)
        x_cell = tf.reshape(x_cell, [1, -1, 1])  # 和上面[H*W,num_anchors,num_class+5]对应
        y_cell = tf.reshape(y_cell, [1, -1, 1])

        # decode
        bbox_x = (x_cell + xy_offset[:, :, :, 0]) / 13
        bbox_y = (y_cell + xy_offset[:, :, :, 1]) / 13
        bbox_w = (self.anchor_size[:, 0] * wh_offset[:, :, :, 0]) / 13
        bbox_h = (self.anchor_size[:, 1] * wh_offset[:, :, :, 1]) / 13

        bboxes = tf.stack([bbox_x - bbox_w / 2, bbox_y - bbox_h / 2, bbox_x + bbox_w / 2, bbox_y + bbox_h / 2],
                          axis=3)

        return bboxes, obj_probs, class_probs

    #将边界框超出整张图片(0,0)—(415,415)的部分cut掉
    def bboxes_cut(self,bbox_min_max, bboxes):
        bboxes = np.copy(bboxes)
        bboxes = np.transpose(bboxes)
        bbox_min_max = np.transpose(bbox_min_max)
        # cut the box
        bboxes[0] = np.maximum(bboxes[0], bbox_min_max[0])  # xmin
        bboxes[1] = np.maximum(bboxes[1], bbox_min_max[1])  # ymin
        bboxes[2] = np.minimum(bboxes[2], bbox_min_max[2])  # xmax
        bboxes[3] = np.minimum(bboxes[3], bbox_min_max[3])  # ymax
        bboxes = np.transpose(bboxes)
        return bboxes

    def bboxes_sort(self,classes, scores, bboxes, top_k=400):
        index = np.argsort(-scores)
        classes = classes[index][:top_k]
        scores = scores[index][:top_k]
        bboxes = bboxes[index][:top_k]
        return classes, scores, bboxes


    def bboxes_iou(self,bboxes1, bboxes2):
        bboxes1 = np.transpose(bboxes1)
        bboxes2 = np.transpose(bboxes2)

        int_ymin = np.maximum(bboxes1[0], bboxes2[0])
        int_xmin = np.maximum(bboxes1[1], bboxes2[1])
        int_ymax = np.minimum(bboxes1[2], bboxes2[2])
        int_xmax = np.minimum(bboxes1[3], bboxes2[3])

        int_h = np.maximum(int_ymax - int_ymin, 0.)
        int_w = np.maximum(int_xmax - int_xmin, 0.)

        # 计算IOU
        int_vol = int_h * int_w  # 交集面积
        vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])  # bboxes1面积
        vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])  # bboxes2面积
        IOU = int_vol / (vol1 + vol2 - int_vol)  # IOU=交集/并集
        return IOU

    # NMS，或者用tf.image.non_max_suppression
    def bboxes_nms(self,classes, scores, bboxes, nms_threshold=0.2):
        keep_bboxes = np.ones(scores.shape, dtype=np.bool)
        for i in range(scores.size - 1):
            if keep_bboxes[i]:
                overlap = self.bboxes_iou(bboxes[i], bboxes[(i + 1):])
                keep_overlap = np.logical_or(overlap < nms_threshold,
                                             classes[(i + 1):] != classes[i])  # IOU没有超过0.5或者是不同的类则保存下来
                keep_bboxes[(i + 1):] = np.logical_and(keep_bboxes[(i + 1):], keep_overlap)

        idxes = np.where(keep_bboxes)
        return classes[idxes], scores[idxes], bboxes[idxes]

    def postprocess(self,bboxes, obj_probs, class_probs, image_shape=(416, 416), threshold=0.5):

        bboxes = np.reshape(bboxes, [-1, 4])
        # 将所有box还原成图片中真实的位置
        bboxes[:, 0:1] *= float(image_shape[1])
        bboxes[:, 1:2] *= float(image_shape[0])
        bboxes[:, 2:3] *= float(image_shape[1])
        bboxes[:, 3:4] *= float(image_shape[0])
        bboxes = bboxes.astype(np.int32)  # 转int


        bbox_min_max = [0, 0, image_shape[1] - 1, image_shape[0] - 1]
        bboxes = self.bboxes_cut(bbox_min_max, bboxes)


        obj_probs = np.reshape(obj_probs, [-1])  # 13*13*5
        class_probs = np.reshape(class_probs, [len(obj_probs), -1])  # (13*13*5,80)
        class_max_index = np.argmax(class_probs, axis=1)  # max类别概率对应的index
        class_probs = class_probs[np.arange(len(obj_probs)), class_max_index]
        scores = obj_probs * class_probs  # 置信度*max类别概率=类别置信度scores

        # 类别置信度scores>threshold的边界框bboxes留下
        keep_index = scores > threshold
        class_max_index = class_max_index[keep_index]
        scores = scores[keep_index]
        bboxes = bboxes[keep_index]

        # (2)排序top_k(默认为400)
        class_max_index, scores, bboxes = self.bboxes_sort(class_max_index, scores, bboxes)
        # (3)NMS
        class_max_index, scores, bboxes = self.bboxes_nms(class_max_index, scores, bboxes)
        return bboxes, scores, class_max_index



    def preprocess_image(self,image, image_size=(416, 416)):

        image_cp = np.copy(image).astype(np.float32)
        image_rgb = cv2.cvtColor(image_cp, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, image_size)
        image_normalized = image_resized.astype(np.float32) / 225.0
        image_expanded = np.expand_dims(image_normalized, axis=0)
        return image_expanded

    def draw_detection(self,im, bboxes, scores, cls_inds, labels):

        imgcv = np.copy(im)
        h, w, _ = imgcv.shape
        for i, box in enumerate(bboxes):
            cls_indx = cls_inds[i]
            thick = int((h + w) / 1000)
            cv2.rectangle(imgcv, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), thick)
            mess = '%s: %.3f' % (labels[cls_indx], scores[i])
            text_loc = (box[0], box[1] - 10)
            cv2.putText(imgcv, mess, text_loc, cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * h, (0, 0, 255), thick)
        # return imgcv
        cv2.imshow("detection_results", imgcv)  # 显示图片
        cv2.waitKey(0)






    '''
    train part
    '''


    def preprocess_true_boxes(self,true_box,anchors,img_size = (416,416)):
        '''
        :param true_box:实际框的位置和类别,2D TENSOR:(batch,5)

        :param anchors:anchors : 实际anchor boxes 的值，论文中使用了五个。[w,h]，都是相对于gird cell 的比值。
                2d
            第二个维度：[w,h]，w,h,都是相对于gird cell长宽的比值。
           [1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]
        :param img_size:
        :return:
           -detectors_mask: 取值是0或者1，这里的shape是[13,13,5,1]

                第四个维度：0/1。1的就是用于预测改true boxes 的 anchor boxes
           -matching_true_boxes:这里的shape是[13,13,5,5]。

        '''
        w,h = img_size
        feature_w = w // 32
        feature_h = h // 32

        num_box_params = true_box.shape[1]
        detectors_mask = np.zeros((feature_h,feature_w,self.num_anchors,1),dtype=np.float32)
        matching_true_boxes = np.zeros((feature_h,feature_w,self.num_anchors,num_box_params),dtype=np.float32)

        for i in true_box:
            #提取类别信息，属于哪类
            box_class = i[4:5]
            #换算成相对于gird cell的值
            box = i[0:4] * np.array([feature_w, feature_h, feature_w, feature_h])
            k = np.floor(box[1]).astype('int') #y方向上属于第几个gird cell
            j = np.floor(box[0]).astype('int') #x方向上属于第几个gird cell
            best_iou = 0
            best_anchor = 0

            #计算anchor boxes 和 true boxes的iou ，一个true box一个best anchor
            for m,anchor in enumerate(anchors):
                box_maxes = box[2:4] / 2.
                box_mins = -box_maxes
                anchor_maxes = (anchor / 2.)
                anchor_mins = -anchor_maxes

                intersect_mins = np.maximum(box_mins, anchor_mins)
                intersect_maxes = np.minimum(box_maxes, anchor_maxes)
                intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
                intersect_area = intersect_wh[0] * intersect_wh[1]
                box_area = box[2] * box[3]
                anchor_area = anchor[0] * anchor[1]
                iou = intersect_area / (box_area + anchor_area - intersect_area)
                if iou > best_iou:
                    best_iou = iou
                    best_anchor = m

            if best_iou > 0:
                detectors_mask[k, j, best_anchor] = 1

                adjusted_box = np.array(  #找到最佳预测anchor boxes
                    [
                        box[0] - j, box[1] - k, #'x,y都是相对于gird cell的位置，左上角[0,0]，右下角[1,1]'
                        np.log(box[2] / anchors[best_anchor][0]), #'对应实际框w,h和anchor boxes w,h的比值取log函数'
                        np.log(box[3] / anchors[best_anchor][1]), box_class #'class实际框的物体是属于第几类'
                    ],
                    dtype=np.float32)
                matching_true_boxes[k, j, best_anchor] = adjusted_box
            return detectors_mask, matching_true_boxes



    def yolo_head(self,feature_map, anchors, num_classes):
        '''
        这个函数是输入yolo的输出层的特征，转化成相对于gird cell坐标的x,y，相对于gird cell长宽的w,h，
        pred_confidence是判断否存在物体的概率，pred_class_prob是sofrmax后各个类别分别的概率
        :param feats:  网络最后一层输出 [none,13,13,125]/[none,13,13,425]
        :param anchors:[5,n]
        :param num_classes:类别数
        :return:x,y,w,h在loss function中计算iou，然后计算iou损失。
                然后和pred_confidence计算confidence_loss，pred_class_prob用于计算classification_loss。
                box_xy : 每张图片的每个gird cell中的每个pred_boxes中心点x,y相对于其所在gird cell的坐标值，左上顶点为[0,0],右下顶点为[1,1]。
                shape:[-1,13,13,5,2].

                box_wh : 每张图片的每个gird cell中的每个pred_boxes的w,h都是相对于gird cell的比值
                shape:[-1,13,13,5,2].

                box_confidence : 每张图片的每个gird cell中的每个pred_boxes的，判断是否存在可检测物体的概率。
                shape:[-1,13,13,5,1]。各维度信息同上。

                box_class_pred : 每张图片的每个gird cell中的每个pred_boxes所框起来的各个类别分别的概率(经过了softmax)。
                shape:[-1,13,13,5,20/80]
'''
        anchors = tf.reshape(tf.constant(anchors,dtype=tf.float32),[1,1,1,self.num_anchors,2])
        num_gird_cell = tf.shape(feature_map)[1:3] #[13,13]
        conv_height_index = K.arange(0,stop=num_gird_cell[0])
        conv_width_index = K.arange(0,stop=num_gird_cell[1])

        conv_height_index = tf.tile(conv_height_index, [num_gird_cell[1]])

        conv_width_index = tf.tile(
            tf.expand_dims(conv_width_index, 0), [num_gird_cell[0], 1])
        conv_width_index = K.flatten(K.transpose(conv_width_index))
        conv_index = K.transpose(K.stack([conv_height_index,conv_width_index]))
        conv_index = K.reshape(conv_index,[1,num_gird_cell[0],num_gird_cell[1],1,2])#[1，13，13，1，2]
        conv_index = K.cast(conv_index,K.dtype(feature_map))
        #[[0,0][0,1]....[0,12],[1,0]...]
        feature_map = K.reshape(feature_map,[-1,num_gird_cell[0],num_gird_cell[1],self.num_anchors,self.num_class + 5])
        num_gird_cell = K.cast(K.reshape(num_gird_cell,[1,1,1,1,2]),K.dtype(feature_map))

        box_xy = K.sigmoid(feature_map[...,:2])
        box_wh = K.exp(feature_map[...,2:4])
        confidence = K.sigmoid(feature_map[...,4:5])
        cls_prob = K.softmax(feature_map[...,5:])

        xy = (box_xy + conv_index) / num_gird_cell
        wh = box_wh * anchors / num_gird_cell

        return xy,wh,confidence,cls_prob



    def loss(self,
             net,
             true_boxes,
             detectors_mask,
             matching_true_boxes,
             anchors,
             num_classes):
        '''
        IOU损失，分类损失，坐标损失

        confidence_loss：
                共有845个anchor_boxes，与true_boxes匹配的用于预测pred_boxes，
                未与true_boxes匹配的anchor_boxes用于预测background。在未与true_boxes匹配的anchor_boxes中，
                与true_boxes的IOU小于0.6的被标记为background，这部分预测正确，未造成损失。
                但未与true_boxes匹配的anchor_boxes中，若与true_boxes的IOU大于0.6的我们需要计算其损失，
                因为它未能准确预测background，与true_boxes重合度过高，就是no_objects_loss。
                而objects_loss则是与true_boxes匹配的anchor_boxes的预测误差。与YOLOv1不同的是修正系数的改变，
                YOLOv1中no_objects_loss和objects_loss分别是0.5和1，而YOLOv2中则是1和5。

        classification_loss:
                经过softmax（）后，20维向量（数据集中分类种类为20种）的均方误差。

        coordinates_loss：
                计算x,y的误差由相对于整个图像（416x416）的offset坐标误差的均方改变为相对于gird cell的offset（这个offset是取sigmoid函数得到的处于（0,1）的值）坐标误差的均方。
                也将修正系数由5改为了1 。计算w,h的误差由w,h平方根的差的均方误差变为了，
                w,h与对true_boxes匹配的anchor_boxes的长宽的比值取log函数，
                和YOLOv1的想法一样，对于相等的误差值，降低对大物体误差的惩罚，加大对小物体误差的惩罚。同时也将修正系数由5改为了1。

        :param net:[batch_size,13,13,125],网络最后一层输出
        :param true_boxes:实际框的位置和类别 [batch,5]
        :param detectors_mask:取值是0或者1，[ batch_size，13,13,5,1]
                1的就是用于预测改true boxes 的 anchor boxes
        :param matching_true_boxes:[-1,13,13,5,5]
        :param anchors:
        :param num_classes:20
        :return:
        '''

        xy, wh, confidence, cls_prob = self.yolo_head(net,anchors,num_classes)
        shape = tf.shape(net)
        feature_map = tf.reshape(net,[-1,shape[1],shape[2],self.num_anchors,num_classes + 5])
        #用于和matching_true_boxes计算坐标损失
        pred_box = tf.concat([K.sigmoid(feature_map[...,0:2]),feature_map[...,2:4]],axis=-1)

        pred_xy = tf.to_float(tf.expand_dims(xy,4))#[-1,13,13,5,2]-->[-1,13,13,5,1,2]
        pred_wh = tf.to_float(tf.expand_dims(wh,4))

        pred_min = tf.to_float(pred_xy - pred_wh / 2.0)
        pred_max = tf.to_float(pred_xy + pred_wh / 2.0)

        true_box_shape = K.shape(true_boxes)
        print(true_box_shape)
        true_boxes = K.reshape(true_boxes,[-1,1,1,1,true_box_shape[1], 5])
        #[-1,1,1,1,-1,5],batch, conv_height, conv_width, num_anchors, num_true_boxes, box_params'

        true_xy = tf.to_float(true_boxes[...,0:2])
        true_wh = tf.to_float(true_boxes[...,2:4])
        true_min = tf.to_float(true_xy - true_wh / 2.0)
        true_max = tf.to_float(true_xy + true_wh / 2.0)

        #计算所以abox和tbox的iou
        intersect_mins = tf.maximum(pred_min, true_min)
        intersect_maxes = tf.minimum(pred_max, true_max)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = tf.to_float(intersect_wh[..., 0] * intersect_wh[..., 1])
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = intersect_areas / union_areas


        #可能会有多个tbox落在同一个cell ，只去iou最大的
        # tf.argmax(iou_scores,4)
        best_ious = K.max(iou_scores, axis=4)
        best_ious = tf.expand_dims(best_ious,axis=-1)

        #选出IOU大于0.6的，若IOU小于0.6的被标记为background，
        obj_dec = tf.cast(best_ious > 0.6,dtype=K.dtype(best_ious))


        #IOU loss
        no_obj_w = (self.no_object_scale * obj_dec * detectors_mask) #
        no_obj_loss = no_obj_w * tf.square(-confidence)
        obj_loss = self.object_scale * detectors_mask * tf.square(1 - confidence)
        confidence_loss = no_obj_loss + obj_loss


        #class loss
        match_cls = tf.cast(matching_true_boxes[...,4],dtype=tf.int32)
        match_cls = tf.one_hot(match_cls,num_classes)

        class_loss = (self.class_scale * detectors_mask * tf.square(match_cls - cls_prob))

        #坐标loss
        match_box = matching_true_boxes[...,0:4]
        coord_loss = self.coordinates_scale * detectors_mask * tf.square(match_box - pred_box)


        confidence_loss_sum = K.sum(confidence_loss)
        class_loss_sum = K.sum(class_loss)
        coord_loss_sum = K.sum(coord_loss)
        all_loss = 0.5 * (confidence_loss_sum + class_loss_sum + coord_loss_sum)

        return all_loss




#v1 - v2 , v2 - v3
# 1、加入BN层 批次归一化   input --> 均值为0方差为1正太分布
#    ---》白化  --> 对‘input 变换到 均值0单位方差内的分布
# #使用：input * w -->bn

if __name__ == '__main__':
    network = yolov2('coco')
    net,x = network.darknet()
    bboxes, obj_probs, class_probs = network.decode(net)
    img = cv2.imread('../../road.jpg')
    shape = img.shape[:2]
    img_r = network.preprocess_image(img)
    saver = tf.train.Saver()
    ckpt_path = '../../model/yolo/v2/yolo2_coco.ckpt'
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,ckpt_path)
    bboxes, obj_probs, class_probs = sess.run([bboxes, obj_probs, class_probs],feed_dict={x:img_r})

    bboxes, scores, class_max_index = network.postprocess(bboxes, obj_probs, class_probs)

    img_detection = network.draw_detection(cv2.resize(img,(416,416)), bboxes, scores, class_max_index, network.CLASS)


    print('done')




'''
 yi、
    第一大层  :conv maxpoiling
    第2大层:3个卷积，maxpool
    3:3个卷积，maxpool
    4：3卷积，maxpool
    5:5卷积，maxpool   -----------
    6:5卷积                       | + add
    7三个卷积---------------------
    conv  
 er:
    ahchors生成和decode
 san:
    裁剪、选出前TOP_K，NMS 
'''
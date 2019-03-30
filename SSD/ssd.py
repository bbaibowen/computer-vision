import tensorflow as tf
import numpy as np
import cv2
import datetime


class ssd(object):

    def __init__(self):
        self.feature_map_size = [[38, 38], [19, 19], [10, 10], [5, 5], [3, 3], [1, 1]]
        self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]
        self.feature_layers = ['block4', 'block7', 'block8', 'block9', 'block10', 'block11']
        self.img_size = (300,300)
        self.num_classes = 21
        self.boxes_len = [4,6,6,6,4,4]
        self.isL2norm = [True,False,False,False,False,False]
        self.anchor_sizes = [[21., 45.], [45., 99.], [99., 153.],[153., 207.],[207., 261.], [261., 315.]]
        self.anchor_ratios = [[2, .5], [2, .5, 3, 1. / 3], [2, .5, 3, 1. / 3],
                         [2, .5, 3, 1. / 3], [2, .5], [2, .5]]
        # self.anchor_steps = [8, 16, 32, 64, 100, 300]
        self.anchor_steps = [8, 16, 30, 60, 100, 300]
        self.prior_scaling = [0.1, 0.1, 0.2, 0.2] #特征图先验框缩放比例
        self.n_boxes = [5776,2166,600,150,36,4]  #8732个
        self.threshold = 0.25

###########    ssd网络架构部分
    def l2norm(self,x, trainable=True, scope='L2Normalization'):
        n_channels = x.get_shape().as_list()[-1]  # 通道数
        l2_norm = tf.nn.l2_normalize(x, dim=[3], epsilon=1e-12)  # 只对每个像素点在channels上做归一化
        with tf.variable_scope(scope):
            gamma = tf.get_variable("gamma", shape=[n_channels, ], dtype=tf.float32,
                                    trainable=trainable)
        return l2_norm * gamma

    def conv2d(self,x,filter,k_size,stride=[1,1],padding='same',dilation=[1,1],activation=tf.nn.relu,scope='conv2d'):
        return tf.layers.conv2d(inputs=x, filters=filter, kernel_size=k_size,
                            strides=stride, dilation_rate=dilation, padding=padding,
                            name=scope, activation=activation)

    def max_pool2d(self,x, pool_size, stride, scope='max_pool2d'):
        return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, name=scope, padding='same')

    def pad2d(self,x, pad):
        return tf.pad(x, paddings=[[0, 0], [pad, pad], [pad, pad], [0, 0]])

    def dropout(self,x, d_rate=0.5):
        return tf.layers.dropout(inputs=x, rate=d_rate)

    def ssd_prediction(self, x, num_classes, box_num, isL2norm, scope='multibox'):
        reshape = [-1] + x.get_shape().as_list()[1:-1]  # 去除第一个和最后一个得到shape
        with tf.variable_scope(scope):
            if isL2norm:
                x = self.l2norm(x)
                print(x)
            # #预测位置  --》 坐标和大小  回归
            location_pred = self.conv2d(x, filter=box_num * 4, k_size=[3,3], activation=None,scope='conv_loc')
            location_pred = tf.reshape(location_pred, reshape + [box_num, 4])
            # 预测类别   --> 分类 sofrmax
            class_pred = self.conv2d(x, filter=box_num * num_classes, k_size=[3,3], activation=None, scope='conv_cls')
            class_pred = tf.reshape(class_pred, reshape + [box_num, num_classes])
            print(location_pred, class_pred)
            return location_pred, class_pred



    def set_net(self,x=None):

        check_points = {}
        predictions = []
        locations = []
        logit = []
        x_re = False

        if x is None:
            x = tf.placeholder(dtype=tf.float32,shape=[None,300,300,3])
            x_re = True

        with tf.variable_scope('ssd_300_vgg'):
            #b1
            net = self.conv2d(x,filter=64,k_size=[3,3],scope='conv1_1')
            net = self.conv2d(net,64,[3,3],scope='conv1_2')
            net = self.max_pool2d(net,pool_size=[2,2],stride=[2,2],scope='pool1')
            #b2
            net = self.conv2d(net, filter=128, k_size=[3, 3], scope='conv2_1')
            net = self.conv2d(net, 128, [3, 3], scope='conv2_2')
            net = self.max_pool2d(net, pool_size=[2, 2], stride=[2, 2], scope='pool2')
            #b3
            net = self.conv2d(net, filter=256, k_size=[3, 3], scope='conv3_1')
            net = self.conv2d(net, 256, [3, 3], scope='conv3_2')
            net = self.conv2d(net, 256, [3, 3], scope='conv3_3')
            net = self.max_pool2d(net, pool_size=[2, 2], stride=[2, 2], scope='pool3')
            #b4
            net = self.conv2d(net, filter=512, k_size=[3, 3], scope='conv4_1')
            net = self.conv2d(net, 512, [3, 3], scope='conv4_2')
            net = self.conv2d(net, 512, [3, 3], scope='conv4_3')
            print(net)
            check_points['block4'] = net
            net = self.max_pool2d(net, pool_size=[2, 2], stride=[2, 2], scope='pool4')
            print('pool4', net)
            #b5
            net = self.conv2d(net, filter=512, k_size=[3, 3], scope='conv5_1')
            net = self.conv2d(net, 512, [3, 3], scope='conv5_2')
            net = self.conv2d(net, 512, [3, 3], scope='conv5_3')
            print('conv5_3',net)
            net = self.max_pool2d(net, pool_size=[3, 3], stride=[1, 1], scope='pool5')
            print('pool5',net)
            #b6
            net = self.conv2d(net,1024,[3,3],dilation=[6,6],scope='conv6')
            print('conv6',net)
            #b7
            net = self.conv2d(net,1024,[1,1],scope='conv7')
            print('conv7',net)
            check_points['block7'] = net
            #b8],scope='conv8_1x1')
            net = self.conv2d(net, 256, [1, 1], scope='conv8_1x1')
            print('conv8_3',net)
            net = self.conv2d(self.pad2d(net, 1), 512, [3, 3], [2, 2], scope='conv8_3x3', padding='valid')
            check_points['block8'] = net
            #b9
            net = self.conv2d(net, 128, [1, 1], scope='conv9_1x1')
            net = self.conv2d(self.pad2d(net,1), 256, [3, 3], [2, 2], scope='conv9_3x3', padding='valid')
            check_points['block9'] = net
            #b10
            net = self.conv2d(net, 128, [1, 1], scope='conv10_1x1')
            net = self.conv2d(net, 256, [3, 3], scope='conv10_3x3', padding='valid')
            check_points['block10'] = net
            #b11
            net = self.conv2d(net, 128, [1, 1], scope='conv11_1x1')
            net = self.conv2d(net, 256, [3, 3], scope='conv11_3x3', padding='valid')
            check_points['block11'] = net
            for i,j in enumerate(self.feature_layers):
                loc,cls = self.ssd_prediction(
                                    x = check_points[j],
                                    num_classes = self.num_classes,
                                    box_num = self.boxes_len[i],
                                    isL2norm = self.isL2norm[i],
                                    scope = j + '_box'
                                    )
                logit.append(cls)
                predictions.append(tf.nn.softmax(cls))
                locations.append(loc)
            if x_re:
                return locations, predictions,x
            else:
                return locations,predictions,logit

###########    ssd网络架构部分结束

##########    先验框部分开始

    #先验框生成
    def ssd_anchor_layer(self,img_size,feature_map_size,anchor_size,anchor_ratio,anchor_step,box_num,offset=0.5):

        y,x = np.mgrid[0:feature_map_size[0],0:feature_map_size[1]]

        y = (y.astype(np.float32) + offset) * anchor_step /img_size[0]
        x = (x.astype(np.float32) + offset) * anchor_step /img_size[1]

        y = np.expand_dims(y,axis=-1)
        x = np.expand_dims(x,axis=-1)
        #计算两个长宽比为1的h、w

        h = np.zeros((box_num,),np.float32)
        w = np.zeros((box_num,),np.float32)

        h[0] = anchor_size[0] /img_size[0]
        w[0] = anchor_size[0] /img_size[0]
        h[1] = (anchor_size[0] * anchor_size[1]) ** 0.5 / img_size[0]
        w[1] = (anchor_size[0] * anchor_size[1]) ** 0.5 / img_size[1]


        for i,j in enumerate(anchor_ratio):
            h[i + 2] = anchor_size[0] / img_size[0] / (j ** 0.5)
            w[i + 2] = anchor_size[0] / img_size[1] * (j ** 0.5)

        return y,x,h,w

    #解码网络
    def ssd_decode(self,location,box,prior_scaling):
        y_a, x_a, h_a, w_a = box

        cx = location[:, :, :, :, 0] * w_a * prior_scaling[0] + x_a  #########################
        cy = location[:, :, :, :, 1] * h_a * prior_scaling[1] + y_a
        w = w_a * tf.exp(location[:, :, :, :, 2] * prior_scaling[2])
        h = h_a * tf.exp(location[:, :, :, :, 3] * prior_scaling[3])


        bboxes = tf.stack([cy - h / 2.0, cx - w / 2.0, cy + h / 2.0, cx + w / 2.0], axis=-1)

        return bboxes


    #先验框筛选
    def choose_anchor_boxes(self, predictions, anchor_box, n_box):
        anchor_box = tf.reshape(anchor_box, [n_box, 4])
        prediction = tf.reshape(predictions, [n_box, 21])
        prediction = prediction[:, 1:]
        classes = tf.argmax(prediction, axis=1) + 1
        scores = tf.reduce_max(prediction, axis=1)


        filter_mask = scores > self.threshold
        classes = tf.boolean_mask(classes, filter_mask)
        scores = tf.boolean_mask(scores, filter_mask)
        anchor_box = tf.boolean_mask(anchor_box, filter_mask)

        return classes, scores, anchor_box

########## 先验框部分结束

######### 训练部分开始

    def bboxes_sort(self,classes, scores, bboxes, top_k=400):
        idxes = np.argsort(-scores)
        classes = classes[idxes][:top_k]
        scores = scores[idxes][:top_k]
        bboxes = bboxes[idxes][:top_k]
        return classes, scores, bboxes

    # 计算IOU
    def bboxes_iou(self,bboxes1, bboxes2):
        bboxes1 = np.transpose(bboxes1)
        bboxes2 = np.transpose(bboxes2)
        # 计算两个box的交集：交集左上角的点取两个box的max，交集右下角的点取两个box的min
        int_ymin = np.maximum(bboxes1[0], bboxes2[0])
        int_xmin = np.maximum(bboxes1[1], bboxes2[1])
        int_ymax = np.minimum(bboxes1[2], bboxes2[2])
        int_xmax = np.minimum(bboxes1[3], bboxes2[3])

        # 计算两个box交集的wh：如果两个box没有交集，那么wh为0(按照计算方式wh为负数，跟0比较取最大值)
        int_h = np.maximum(int_ymax - int_ymin, 0.)
        int_w = np.maximum(int_xmax - int_xmin, 0.)

        # 计算IOU
        int_vol = int_h * int_w  # 交集面积
        vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])  # bboxes1面积
        vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])  # bboxes2面积
        iou = int_vol / (vol1 + vol2 - int_vol)  # IOU=交集/并集
        return iou

    # NMS
    def bboxes_nms(self,classes, scores, bboxes, nms_threshold=0.5):
        keep_bboxes = np.ones(scores.shape, dtype=np.bool)
        for i in range(scores.size - 1):
            if keep_bboxes[i]:
                overlap = self.bboxes_iou(bboxes[i], bboxes[(i + 1):])
                keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i + 1):] != classes[i])
                keep_bboxes[(i + 1):] = np.logical_and(keep_bboxes[(i + 1):], keep_overlap)
        idxes = np.where(keep_bboxes)
        return classes[idxes], scores[idxes], bboxes[idxes]


######## 训练部分结束

    def handle_img(self,img_path):
        means = np.array((123., 117., 104.))
        self.img = cv2.imread(img_path)
        # img = self.img
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) - means
        # img = cv2.resize(img,self.img_size)
        # img = np.expand_dims(img,axis=0)
        img = np.expand_dims(cv2.resize(cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB) - means,self.img_size),axis=0)
        return img


    def draw_rectangle(self,img, classes, scores, bboxes, colors, thickness=2):
        shape = img.shape
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            # color = colors[classes[i]]
            p1 = (int(bbox[0] * shape[0]), int(bbox[1] * shape[1]))
            p2 = (int(bbox[2] * shape[0]), int(bbox[3] * shape[1]))
            cv2.rectangle(img, p1[::-1], p2[::-1], colors[1], 1)
            # Draw text...
            s = '%s/%.3f' % (self.classes[classes[i] - 1], scores[i])
            p1 = (p1[0] - 5, p1[1])
            cv2.putText(img, s, p1[::-1], cv2.FONT_HERSHEY_DUPLEX, 0.5, colors[1], 1)
        # cv2.namedWindow("ssd", 0);
        # cv2.resizeWindow("ssd", 640, 480);
        cv2.imshow('ssd', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




    def run_this(self,locations,predictions):

        layers_anchors = []
        classes_list = []
        scores_list = []
        bboxes_list = []
        for i, s in enumerate(self.feature_map_size):
            anchor_bboxes = self.ssd_anchor_layer(self.img_size, s,
                                                  self.anchor_sizes[i],
                                                  self.anchor_ratios[i],
                                                  self.anchor_steps[i],
                                                  self.boxes_len[i])
            layers_anchors.append(anchor_bboxes)
        for i in range(len(predictions)):
            d_box = self.ssd_decode(locations[i], layers_anchors[i], self.prior_scaling)
            cls, sco, box = self.choose_anchor_boxes(predictions[i], d_box, self.n_boxes[i])
            classes_list.append(cls)
            scores_list.append(sco)
            bboxes_list.append(box)
        classes = tf.concat(classes_list, axis=0)
        scores = tf.concat(scores_list, axis=0)
        bboxes = tf.concat(bboxes_list, axis=0)
        return classes,scores,bboxes


    def bboxes_encode(self, labels, bboxes, anchors, scope=None):

        target_labels, target_localizations, target_scores = tf_ssd_bboxes_encode(
                                                                labels, bboxes, anchors,
                                                                self.num_classes,
                                                                prior_scaling=self.prior_scaling,
                                                                scope=scope)

        return target_labels, target_localizations, target_scores

    def smooth_L1(self,x):
        absx = tf.abs(x)
        minx = tf.minimum(absx, 1)
        r = 0.5 * ((absx - 1) * minx + absx)
        return r

    def ssd_losses(self,logits, localisations,  # 预测类别，位置
                   gclasses, glocalisations, gscores,  # ground truth类别，位置，得分
                   match_threshold=0.5,
                   negative_ratio=3.,
                   alpha=1.,
                   scope=None):

        '''
            n_neg就是负样本的数量，negative_ratio正负样本比列，默认就是3, 后面的第一个取最大，我觉得是保证至少有负样本，
            max_neg_entries这个就是负样本的数量，n_neg = tf.minimum(n_neg, max_neg_entries)，这个比较很好理解，万一
            你总样本比你三倍正样本少，所以需要选择小的，所以这个地方保证足够的负样本，nmask表示我们所选取的负样本，
            tf.nn.top_k，这个是选取前k = neg个负例，因为取了负号，表示选择的交并比最小的k个，minval就是选择负例里面交并比
            最大的，nmask就是把我们选择的负样例设为整数，就是提取出我们选择的，tf.logical_and就是同时为真，首先。需要是
            负例，其次值需要大于minval，因为取了负数，所以nmask就是我们所选择的负例，fnmask就是就是我们选取的负样本只是
            数据类型变了，由bool变为了浮点型，(dtype默认是浮点型)
        '''



        with tf.name_scope(scope, 'ssd_losses'):
            # 提取类别数和batch_size # tensor_shape函数可以取代
            num_classes = int(logits[0].shape[-1])
            batch_size = int(logits[0].shape[0])
            print('num_cl,batch',num_classes,batch_size)
            # Flatten out all vectors!
            flogits = []
            fgclasses = []
            fgscores = []
            flocalisations = []
            fglocalisations = []
            for i in range(len(logits)):  # 按照图片循环
                flogits.append(tf.reshape(logits[i], [-1, num_classes]))
                fgclasses.append(tf.reshape(gclasses[i], [-1]))
                fgscores.append(tf.reshape(gscores[i], [-1]))
                flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
                fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))

            # And concat the crap!

            # logits (none,38,38,4,21)

            logits = tf.concat(flogits, axis=0)  # 全部的搜索框，对应的21类别的输出
            gclasses = tf.concat(fgclasses, axis=0)  # 全部的搜索框，真实的类别数字
            gscores = tf.concat(fgscores, axis=0)  # 全部的搜索框，和真实框的IOU
            localisations = tf.concat(flocalisations, axis=0)
            glocalisations = tf.concat(fglocalisations, axis=0)

            dtype = logits.dtype

            pmask = gscores > match_threshold  # (全部搜索框数目, 21)，类别搜索框和真实框IOU大于阈值
            fpmask = tf.cast(pmask, dtype)  # 浮点型前景掩码（前景假定为含有对象的IOU足够的搜索框标号）
            n_positives = tf.reduce_sum(fpmask)  # tp总数
            # Hard negative mining...


            no_classes = tf.cast(pmask, tf.int32)
            predictions = tf.nn.softmax(logits)  # 此时每一行的21个数转化为概率

            nmask = tf.logical_and(tf.logical_not(pmask),
                                   gscores > -0.5)  # IOU达不到阈值的类别搜索框位置记1
            print(nmask)
            fnmask = tf.cast(nmask, dtype)
            nvalues = tf.where(nmask,
                               predictions[:, 0],  # 框内无物体标记为背景预测概率
                               1. - fnmask)  # 框内有物体位置标记为1
            nvalues_flat = tf.reshape(nvalues, [-1])

            # Number of negative entries to select.
            # 在nmask中剔除n_neg个最不可能背景点(对应的class0概率最低)
            max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
            # 3 × tpmask
            n_negs = tf.cast(negative_ratio * n_positives, tf.int32)
            n_neg = tf.minimum(n_negs, max_neg_entries)
            val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)  # 最不可能为背景的n_neg个点
            max_hard_pred = -val[-1]
            # Final negative mask.
            nmask = tf.logical_and(nmask, nvalues < max_hard_pred)  # 不是前景，又最不像背景的n_neg个点
            fnmask = tf.cast(nmask, dtype)

            # Add cross-entropy loss.
            with tf.name_scope('cross_entropy_pos'):
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                      labels=gclasses)  # 0-20

                loss_pos = tf.div(tf.reduce_sum(loss * fpmask), batch_size,name='value')


            with tf.name_scope('cross_entropy_neg'):
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                      labels=no_classes)  # {0,1}
                loss_neg = tf.div(tf.reduce_sum(loss * fnmask), batch_size,name='value')


            # Add localization loss: smooth L1, L2, ...
            with tf.name_scope('localization'):
                # Weights Tensor: positive mask + random negative.
                weights = tf.expand_dims(alpha * fpmask, axis=-1)
                # alpha * fpm = (n,)
                loss = self.smooth_L1(localisations - glocalisations) #(m,n)

                loss_loc = tf.div(tf.reduce_sum(loss * weights), batch_size,name='value')


            return loss_pos,loss_neg,loss_loc

def tf_ssd_bboxes_encode(labels,    #编码：真实值和ANCHOR BOX  、编码
                         bboxes,
                         anchors,
                         num_classes,
                         prior_scaling=(0.1, 0.1, 0.2, 0.2),
                         dtype=tf.float32,
                         scope='ssd_bboxes_encode'):
    with tf.name_scope(scope):
        target_labels = []
        target_localizations = []
        target_scores = []

        for i, anchors_layer in enumerate(anchors):
            with tf.name_scope('bboxes_encode_block_%i' % i):
                # (m,m,k)，xywh(m,m,4k)，(m,m,k)
                t_labels, t_loc, t_scores = tf_ssd_bboxes_encode_layer(labels, bboxes, anchors_layer,
                                               num_classes)
                target_labels.append(t_labels)
                target_localizations.append(t_loc)
                target_scores.append(t_scores)

        return target_labels, target_localizations, target_scores

def tf_ssd_bboxes_encode_layer(labels,  # (n,)
                               bboxes,  # (n, 4)
                               anchors_layer,  # y(m, m, 1), x(m, m, 1), h(k,), w(k,)
                               num_classes,
                               prior_scaling=(0.1, 0.1, 0.2, 0.2),
                               dtype=tf.float32):
    yref, xref, href, wref = anchors_layer

    ymin = yref - href / 2.
    xmin = xref - wref / 2.
    ymax = yref + href / 2.
    xmax = xref + wref / 2.
    vol_anchors = (xmax - xmin) * (ymax - ymin)
    shape = (yref.shape[0], yref.shape[1], href.size)

    feat_labels = tf.zeros(shape, dtype=tf.int64)  # (m, m, k)
    feat_scores = tf.zeros(shape, dtype=dtype)

    feat_ymin = tf.zeros(shape, dtype=dtype)
    feat_xmin = tf.zeros(shape, dtype=dtype)
    feat_ymax = tf.ones(shape, dtype=dtype)
    feat_xmax = tf.ones(shape, dtype=dtype)

    def jaccard_with_anchors(bbox):

        int_ymin = tf.maximum(ymin, bbox[0])  # (m, m, k)
        int_xmin = tf.maximum(xmin, bbox[1])
        int_ymax = tf.minimum(ymax, bbox[2])
        int_xmax = tf.minimum(xmax, bbox[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        # 处理搜索框和bbox之间的联系
        inter_vol = h * w  # 交集面积
        union_vol = vol_anchors - inter_vol \
                    + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])  # 并集面积
        jaccard = tf.div(inter_vol, union_vol)  # 交集/并集，即IOU
        return jaccard  # (m, m, k)




    def condition(i,feat_labels, feat_scores,
             feat_ymin, feat_xmin, feat_ymax, feat_xmax):

        r = tf.less(i, tf.shape(labels))
        return r[0]

    def body(i, feat_labels, feat_scores,
             feat_ymin, feat_xmin, feat_ymax, feat_xmax):
        """
          更新功能标签、分数和bbox。
            -JacCard>0.5时赋值；
        """


        label = labels[i]  # 当前图片上第i个对象的标签
        bbox = bboxes[i]  # 当前图片上第i个对象的真实框bbox

        jaccard = jaccard_with_anchors(bbox)  # 当前对象的bbox和当前层的搜索网格IOU


        mask = tf.greater(jaccard, feat_scores)  # 掩码矩阵，IOU大于历史得分的为True
        mask = tf.logical_and(mask, feat_scores > -0.5)

        imask = tf.cast(mask, tf.int64) #[1,0,1,1,0]
        fmask = tf.cast(mask, dtype)    #[1.,0.,1.,0. ... ]

        # Update values using mask.
        # 保证feat_labels存储对应位置得分最大对象标签，feat_scores存储那个得分
        # (m, m, k) × 当前类别 + (1 - (m, m, k)) × (m, m, k)
        # 更新label记录，此时的imask已经保证了True位置当前对像得分高于之前的对象得分，其他位置值不变
        feat_labels = imask * label + (1 - imask) * feat_labels
        # 更新score记录，mask为True使用本类别IOU，否则不变
        feat_scores = tf.where(mask, jaccard, feat_scores)

        # 下面四个矩阵存储对应label的真实框坐标
        # (m, m, k) × 当前框坐标scalar + (1 - (m, m, k)) × (m, m, k)
        feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
        feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
        feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
        feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax



        return [i + 1, feat_labels, feat_scores,
                feat_ymin, feat_xmin, feat_ymax, feat_xmax]

    i = 0
    (i,
     feat_labels, feat_scores,
     feat_ymin, feat_xmin,
     feat_ymax, feat_xmax) = tf.while_loop(condition, body,
                                           [i,
                                            feat_labels, feat_scores,
                                            feat_ymin, feat_xmin,
                                            feat_ymax, feat_xmax])
    # Transform to center / size.
    # 这里的y、x、h、w指的是对应位置所属真实框的相关属性
    feat_cy = (feat_ymax + feat_ymin) / 2.
    feat_cx = (feat_xmax + feat_xmin) / 2.
    feat_h = feat_ymax - feat_ymin
    feat_w = feat_xmax - feat_xmin

    # Encode features.

    # ((m, m, k) - (m, m, 1)) / (k,) * 10
    # 以搜索网格中心点为参考，真实框中心的偏移，单位长度为网格hw
    feat_cy = (feat_cy - yref) / href / prior_scaling[0]
    feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
    # log((m, m, k) / (m, m, 1)) * 5
    # 真实框宽高/搜索网格宽高，取对
    feat_h = tf.log(feat_h / href) / prior_scaling[2]
    feat_w = tf.log(feat_w / wref) / prior_scaling[3]
    # Use SSD ordering: x / y / w / h instead of ours.(m, m, k, 4)
    feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
    return feat_labels, feat_localizations, feat_scores

'''
只要修改
img = sd.handle_img('tetst.jpg') 这一行代码就好啦，把你想预测的图片放进去
'''


if __name__ == '__main__':
    start_time = datetime.datetime.now()
    sd = ssd()


    locations, predictions, x = sd.set_net()

    classes, scores, bboxes = sd.run_this(locations, predictions)
    sess = tf.Session()
    ckpt_filename = '../../Nn/ssd_vgg_300_weights.ckpt'
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_filename)

    img = sd.handle_img('../road.jpg')

    rclasses, rscores, rbboxes = sess.run([classes, scores, bboxes], feed_dict={x: img})

    rclasses, rscores, rbboxes = sd.bboxes_sort(rclasses, rscores, rbboxes)

    rclasses, rscores, rbboxes = sd.bboxes_nms(rclasses, rscores, rbboxes)
    print(datetime.datetime.now() - start_time)
    sd.draw_rectangle(sd.img,rclasses,rscores,rbboxes,[[0,0,255],[255,0,0]])








import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

slim = tf.contrib.slim


class Faster_RCNN(object):

    def __init__(self):
        self.CLASSES = ['background',
                   'aeroplane', 'bicycle', 'bird', 'boat',
                   'bottle', 'bus', 'car', 'cat', 'chair',
                   'cow', 'diningtable', 'dog', 'horse',
                   'motorbike', 'person', 'pottedplant',
                   'sheep', 'sofa', 'train', 'tvmonitor']
        self.n_class = len(self.CLASSES)
        #这个参数指定了最初的类似感受野的区域大小，因为经过多层卷积池化之后，
        # feature map上一点的感受野对应到原始图像就会是一个区域，这里设置的是16，
        # 也就是feature map上一点对应到原图的大小为16x16的区域。也可以根据需要自己设置。
        self.base_feat = 16
        self.base_size = [0,0,15,15]
        self.anchors_ratios = [.5,1.,2.]
        self.anchors_scale = [8,16,32]
        self.feature_map = (38,50)
        self.num_anchors = 17100 #38*50*9
        self.test_pre_nms_topN = 6000
        self.test_post_nms_topN = 128
        self.nms_threshold = .7
        self.img_shape = (600,800)
        self.pooling_size = 7


    def bulit_vgg_net(self):

        x = tf.placeholder(dtype=tf.float32,shape=[None,600,800,3])

        # Layer  1
        # 224×224×64
        net = slim.conv2d(x,64,[3,3],trainable=False, scope='conv1_1')
        net = slim.conv2d(net, 64, [3, 3], trainable=False, scope='conv1_2')

        # net = slim.repeat(x, 2, slim.conv2d, 64, [3, 3], trainable=False, scope='conv1')
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool1')  # 112×112×64

        # Layer  2
        net = slim.conv2d(net, 128, [3, 3], trainable=False, scope='conv2_1')
        net = slim.conv2d(net, 128, [3, 3], trainable=False, scope='conv2_2')
        # net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], trainable=False, scope='conv2')  # 112×112×128
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool2')  # 56×56×128

        # Layer 3
        net = slim.conv2d(net, 256, [3, 3], trainable=True, scope='conv3_1')
        net = slim.conv2d(net, 256, [3, 3], trainable=True, scope='conv3_2')
        net = slim.conv2d(net, 256, [3, 3], trainable=True, scope='conv3_3')
        # net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], trainable=True, scope='conv3')  # 56×56×256
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool3')  # 28×28×256

        # Layer 4
        net = slim.conv2d(net, 512, [3, 3], trainable=True, scope='conv4_1')
        net = slim.conv2d(net, 512, [3, 3], trainable=True, scope='conv4_2')
        net = slim.conv2d(net, 512, [3, 3], trainable=True, scope='conv4_3')
        # net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=True, scope='conv4')  # 28×28×512
        net = slim.max_pool2d(net, [2, 2], padding='SAME', scope='pool4')  # 14×14×512

        # Layer 5  38,50,512
        net = slim.conv2d(net, 512, [3, 3], trainable=True, scope='conv5_1')
        net = slim.conv2d(net, 512, [3, 3], trainable=True, scope='conv5_2')
        net = slim.conv2d(net, 512, [3, 3], trainable=True, scope='conv5_3')
        # net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], trainable=True, scope='conv5')  # 14×14×512

        return net,x


    def anchor(self):  #个数  38*50*9 = 17100个
        '''
        先设置基础anchor，默认为[0,0,15,15],计算基础anchor的宽(16)和高(16)，anchor中心(7.5,7.5)，以及面积(256)
        计算基础anchor的面积，分别除以[0.5,1,2],得到[512,256,128]
        anchor的宽度w由三个面积的平方根值确定,得到[23,16,11]
        anchor的高度h由[23,16,11]*[0.5,1,2]确定,得到[12,16,22].
        由anchor的中心以及不同的宽和高可以得到此时的anchors.
        :param feature_map:
        :return:
        '''

        #将默认anchor的四个坐标值转化成（宽，高，中心点横坐标，中心点纵坐标）的形式
        # 默认anchor 的值为 宽高16 中心（7.5,7.5）
        w = self.base_size[2] + 1 #16
        h = self.base_size[3] + 1 #16
        x = self.base_size[0] + 0.5 * (w - 1) # 7.5
        y = self.base_size[1] + 0.5 * (h - 1) # 7.5

        #计算基础anchor的面积
        # anchor的宽度w由三个面积的平方根值确定, 得到[23, 16, 11]
        # anchor的高度h由[23, 16, 11] * [0.5, 1, 2] 确定, 得到[12, 16, 22].
        size = w * h
        size_ratios = size / np.array(self.anchors_ratios) #[512,256,128]
        print(size_ratios)
        ws = np.round(np.sqrt(size_ratios))
        print(ws)
        hs = np.round(ws * self.anchors_ratios)


        ws = ws[:, np.newaxis]
        hs = hs[:, np.newaxis]
        print(ws)
        anchors = np.hstack((x - 0.5 * (ws - 1),
                             y - 0.5 * (hs - 1),
                             x + 0.5 * (ws - 1),
                             y + 0.5 * (hs - 1)))
        #扩展
        all_anchors = []
        for i,an in enumerate(anchors):
            w = an[2] - an[0] + 1
            h = an[3] - an[1] + 1
            x_ctr = an[0] + 0.5 * (w - 1)
            y_ctr = an[1] + 0.5 * (an[3] - 1)
            for j in self.anchors_scale:
                ws = w * j
                hs = h * j
                all_anchors.append([x_ctr - 0.5 * (ws - 1),
                                    y_ctr - 0.5 * (hs - 1),
                                    x_ctr + 0.5 * (ws - 1),
                                    y_ctr + 0.5 * (hs - 1)])
        all_anchors = np.array(all_anchors)
        return all_anchors

    def mapping_anchors(self, height,width,feat_stride):
        #上述得到的是没有映射回feature map的anchors，现在要映射回feature map
        #所以height、width是feature map的h w，对应w*h个
        #再×16 就是每个anchors点
        #比如生成的【1,2,3】 ----> 映射回原图就是 [16,32,48]
        anchors = self.anchor()

        x1 = tf.range(width) * feat_stride
        y1 = tf.range(height) * feat_stride
        # x1 = np.arange(0,width) * feat_stride
        # y1 = np.arange(0,height) * feat_stride
        #向量矩阵
        x2,y2 = tf.meshgrid(x1,y1)
        # x2 , y2 = np.meshgrid(x1,y1)
        x2 = tf.reshape(x2,(-1,))
        y2 = tf.reshape(y2,(-1,))
        #合并   拉直
        shifts = tf.transpose(tf.stack([x2, y2, x2, y2]))
        shifts = tf.transpose(
            tf.reshape(shifts, shape=[1, height * width, 4]), perm=(1, 0, 2)
        )
        anchors = anchors.reshape((1,9,4))
        anchors = tf.constant(anchors,tf.int32)

        all_anchors = tf.add(anchors, shifts)

        all_anchors = tf.cast(tf.reshape(all_anchors,[-1,4]),tf.float32)

        return all_anchors

    def _reshape_layer(self, bottom, num_dim, name):
        input_shape = tf.shape(bottom)
        with tf.variable_scope(name) as scope:

            to_caffe = tf.transpose(bottom, [0, 3, 1, 2])

            reshaped = tf.reshape(to_caffe,
                                  tf.concat(axis=0, values=[[1, num_dim, -1], [input_shape[2]]]))

            to_tf = tf.transpose(reshaped, [0, 2, 3, 1])
            return to_tf

    def _softmax_layer(self, bottom, name):
        if name.startswith('rpn_cls_prob_reshape'):
            input_shape = tf.shape(bottom)
            bottom_reshaped = tf.reshape(bottom, [-1, input_shape[-1]])
            reshaped_score = tf.nn.softmax(bottom_reshaped, name=name)
            return tf.reshape(reshaped_score, input_shape)
        return tf.nn.softmax(bottom, name=name)


    def build_rpn(self,net,is_training = False):
        '''
        featureMap的每个点为中心，生成9种不同大小尺度的候选框
        3*3的卷积，不改变图像大小，增加卷积映射区域和空间信息
        1*1的卷积，对以每个点为中心的9种候选框进行前后景分类(n,W,H,18)
        (n,18,W,H)(n,2,9W,H)成[1,9*H,W,2]，便于caffee_softmax进行 fg/bg二分类
        softmax判定单个框的foreground与background分数[1,9*H,W,2]，(n,2,9W,H)(n,18,W,H)转化为(1,H,W,18)，即以每个点为中心的9种候选框前后景分数
        1*1的卷积，对前景区域的位置参数x1,y1,x2,y2进行回归
        anchor_target_layer：得到(bg/fg)的标签值和bbox的偏移标签值
        计算RPN的分类损失rpn_cls_loss和回归框损失rpn_bbox_loss

        生成anchors -> softmax分类器提取fg anchors -> bbox reg回归fg anchors -> Proposal Layer生成proposals

        :param net:
        :return:
        '''


        #RPN第一次卷积

        # rpn = slim.conv2d(net, 512, [3, 3], trainable=is_training, weights_initializer=initializer,
        #                   scope="rpn_conv/3x3")
        # rpn = self.conv2d(net,512,3,1,scope="rpn_conv/3x3")
        rpn = slim.conv2d(net, 512, [3, 3], trainable=is_training,scope="rpn_conv/3x3")
        # print('rpn',rpn)
        # rpn_cls_score = (1,38,50,18)  二分类
        # rpn_cls_score = self.conv2d(rpn,9*2,1,1,scope='rpn_cls_score',padding='VALID',use_relu=False)
        # rpn_bbox_pred = self.conv2d(rpn,9*4,1,1,scope='rpn_bbox_pred',padding='VALID',use_relu=False)
        rpn_cls_score = slim.conv2d(rpn,
                                    9 * 2,
                                    [1, 1],
                                    trainable=is_training,
                                    padding='VALID',
                                    activation_fn=None,
                                    scope='rpn_cls_score')
        rpn_bbox_pred = slim.conv2d(rpn,
                                   9 * 4,
                                   [1,1],
                                   trainable=is_training,
                                   padding='VALID',
                                   activation_fn=None,
                                   scope='rpn_bbox_pred')



        # rpn_cls_score: (n, W, H, 18)
        # < transpose > (n, 18, W, H)
        # < reshape > (n, 2, 9W, H)
        # < transpose > (n, 9W, H, 2)
        # < softmax > (n, 9W, H, 2)
        # < transpose > (n, 2, 9W, H)
        # < reshape > (n, 18, W, H)
        # < transpose > (n, W, H, 18)
        # rpn_cls_score = tf.transpose(rpn_cls_score,(0,3,1,2))
        # rpn_cls_score = tf.reshape(rpn_cls_score,[-1,2,9*38,50])
        # rpn_cls_score = tf.transpose(rpn_cls_score,[0,2,3,1])
        # rpn_cls_prob = tf.nn.softmax(rpn_cls_score)
        # rpn_cls_prob = tf.transpose(rpn_cls_prob, [0, 3, 1, 2])  # (n,2,9W,H)
        # rpn_cls_prob = tf.reshape(rpn_cls_prob, [-1, 18, 38, 50])  # (n,18,W,H)
        # rpn_cls_prob = tf.transpose(rpn_cls_prob, [0, 2, 3, 1])
        rpn_cls_score_reshape = self._reshape_layer(rpn_cls_score,2,'rpn_cls_score_reshape')
        rpn_cls_prob_reshape = self._softmax_layer(rpn_cls_score_reshape, "rpn_cls_prob_reshape")

        rpn_cls_pred = tf.argmax(tf.reshape(rpn_cls_score_reshape, [-1, 2]), axis=1, name="rpn_cls_pred")

        rpn_cls_prob = self._reshape_layer(rpn_cls_prob_reshape, 9 * 2, "rpn_cls_prob") #

        return rpn_cls_prob,rpn_bbox_pred


    #anchors修正
    def correct_anchors(self,anchors,rpn_bbox):
        #rpn_bbox:(n,4)
        #anchors

        boxes = tf.cast(anchors, rpn_bbox.dtype)
        widths = tf.subtract(boxes[:, 2], boxes[:, 0]) + 1.0
        heights = tf.subtract(boxes[:, 3], boxes[:, 1]) + 1.0
        ctr_x = tf.add(boxes[:, 0], widths * 0.5)
        ctr_y = tf.add(boxes[:, 1], heights * 0.5)

        dx = rpn_bbox[:, 0]
        dy = rpn_bbox[:, 1]
        dw = rpn_bbox[:, 2]
        dh = rpn_bbox[:, 3]



        pred_ctr_x = tf.add(tf.multiply(dx, widths), ctr_x)
        pred_ctr_y = tf.add(tf.multiply(dy, heights), ctr_y)
        pred_w = tf.multiply(tf.exp(dw), widths)
        pred_h = tf.multiply(tf.exp(dh), heights)

        pred_boxes0 = tf.subtract(pred_ctr_x, pred_w * 0.5)
        pred_boxes1 = tf.subtract(pred_ctr_y, pred_h * 0.5)
        pred_boxes2 = tf.add(pred_ctr_x, pred_w * 0.5)
        pred_boxes3 = tf.add(pred_ctr_y, pred_h * 0.5)

        return tf.stack([pred_boxes0, pred_boxes1, pred_boxes2, pred_boxes3], axis=1)




    def cilp_bbox(self,bboxs):



        x1 = tf.maximum(tf.minimum(bboxs[:, 0], self.img_shape[1] - 1), 0)
        # y1 >= 0
        y1 = tf.maximum(tf.minimum(bboxs[:, 1], self.img_shape[0] - 1), 0)
        # x2 < im_shape[1]
        x2 = tf.maximum(tf.minimum(bboxs[:, 2], self.img_shape[1] - 1), 0)
        # y2 < im_shape[0]
        y2 = tf.maximum(tf.minimum(bboxs[:, 3], self.img_shape[0] - 1), 0)

        box = tf.stack([x1, y1, x2, y2], axis=1)

        return box

    def nms(self,proposals,score):
        pass



    #供给候选区
    def choose_proposal_layer(self,rpn_prob,rpn_bbox,anchors):
        '''
        这一部分也就是从17100个anchors中选出6000个作为候选框，  #test直接跳过6000
        再经过NMS留下300个box递交给下一层
        :param rpn_prob:  anchors得分
        :param rpn_bbox:  预测的框
        :param anchors:   全部的anchors
        :return:

        '''

        # 1）rpn_cls_prob中第四维度，前9位是背景的概率，后9位是前景的概率，所以首先要取出前景的概率，
        # 即scores = (1, 38, 50, 9) ，之后reshape成(1×38×50×9, 1)即（17100, 1）
        scores = rpn_prob[:, :, :, 9:]  # 取出前景的分数 scores = (1,38,50,9)
        scores = tf.reshape(scores,(-1,))


        #2）将rpn_bbox_pred = （1,38,50,36） reshape成为（1×38×50×9,4），即rpn_bbox_pred=（17100,4）
        rpn_bbox = tf.reshape(rpn_bbox,(-1,4))

        #对所有的anchors进行修正
        proposal = self.correct_anchors(anchors,rpn_bbox)
        #裁剪  ----> 超过img_shape的proposal 裁剪至img 边界
        cilp_proposal = self.cilp_bbox(proposal)


        #4） NMS  注意，第一个参数是box --> 2-d tensor 第二个参数是 score --> 1-d tensor
        nms = tf.image.non_max_suppression(cilp_proposal,
                                           scores,
                                           max_output_size = self.test_post_nms_topN,
                                           iou_threshold = self.nms_threshold)

        #5) 筛选
        proposal_boxes = tf.to_float(tf.gather(cilp_proposal, nms))
        scores = tf.gather(scores, nms)
        scores = tf.reshape(scores, shape=(-1, 1))

        #为要进行roi_pooling，在保留框的坐标信息前面插入batch中图片的编号信息。
        # 此时，由于batch_size为1，因此都插入0
        batch_inds = tf.zeros((tf.shape(nms)[0], 1), dtype=tf.float32)
        blob = tf.concat([batch_inds, proposal_boxes], 1)

        blob.set_shape([None, 5])
        scores.set_shape([None, 1])

        return blob,scores


    #ROI POOLING
    def roi_pooling(self, conv, rois, name):
        '''
        产生目标porposals的坐标后，因为size大小不一样，需要进行ROI池化，使输出为统一维度。
        :param conv:  第五层卷积
        :param rois:  我们筛选出来的proposals
        :param name:
        :return:
        '''
        with tf.variable_scope(name) as scope:
            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            # Get the normalized coordinates of bounding boxes
            bottom_shape = tf.shape(conv)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self.base_feat)
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self.base_feat)
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height
            # Won't be back-propagated to rois anyway, but to save time
            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            pre_pool_size = self.pooling_size * 2
            crops = tf.image.crop_and_resize(conv, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],
                                             name="crops")

        return slim.max_pool2d(crops, [2, 2], padding='SAME')


    #fc
    def vgg_back(self,net):


        flat = slim.flatten(net, scope='flatten')
        fc6 = slim.fully_connected(flat, 4096, scope='fc6')
        fc7 = slim.fully_connected(fc6, 4096, scope='fc7')

        return fc7

    def classification(self,net):



        cls_scores = slim.fully_connected(net,self.n_class,scope = 'cls_score')

        cls_pred = tf.nn.softmax(cls_scores)
        cls_prob = tf.argmax(cls_pred,1)
        self.pro = cls_prob
        bbox_pred = slim.fully_connected(net,self.n_class * 4,scope = 'bbox_pred')





        return cls_pred,bbox_pred

    def handle_img(self,img_path):
        means = np.array((123., 117., 104.))
        img = cv2.imread(img_path)
        # img = cv2.resize(img,(600,800))
        img = np.expand_dims(cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) - means,(600,800)),axis=0)

        return img.transpose([0,2,1,3])



if __name__ == '__main__':
    img_path = '../road.jpg'

    fr = Faster_RCNN()
    img = fr.handle_img(img_path)
    net,x = fr.bulit_vgg_net()
    anchors = fr.mapping_anchors(38,50,16)

    rpn_cls_prob, rpn_bbox_pred = fr.build_rpn(net)

    proposals, scores = fr.choose_proposal_layer(rpn_cls_prob, rpn_bbox_pred, anchors)
    pool5 = fr.roi_pooling(net, proposals, 'pool5')

    fc7 = fr.vgg_back(pool5)
    cls_pred, bbox_pred = fr.classification(fc7)






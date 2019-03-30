import numpy as np
import tensorflow as tf
import cv2





class Yolo(object):

    def __init__(self):

        self.classes = ["aeroplane", "bicycle", "bird", "boat", "bottle",
                        "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant",
                        "sheep", "sofa", "train","tvmonitor"]
        self.C = len(self.classes) # number of classes
        # offset
        self.x_offset = np.transpose(np.reshape(np.array([np.arange(7)]*7*2),
                                              [2, 7, 7]), [1, 2, 0])
        self.y_offset = np.transpose(self.x_offset, [1, 0, 2])
        #x、y shape = (7,7,2)

        self.threshold = 0.2  # confidence scores
        self.iou_threshold = 0.5

        self.max_output_size = 10
        self.img_shape = (448,448)

        self.batch_size = 45

        self.coord_scale = 5.
        self.noobject_scale = 1.
        self.object_scale = 1.
        self.class_scale = 2.




    def leak_relu(self,x, alpha=0.1):
        return tf.maximum(alpha * x, x)


#################  网络部分
    def _build_net(self):

        x = tf.placeholder(tf.float32, [None, 448, 448, 3])
        with tf.variable_scope('yolo'):
            with tf.variable_scope('conv_2'):
                net = self._conv_layer(x,  64, 7, 2,'conv_2')
            net = self._maxpool_layer(net,  2, 2)
            with tf.variable_scope('conv_4'):
                net = self._conv_layer(net,  192, 3, 1,'conv_4')
            net = self._maxpool_layer(net, 2, 2)
            with tf.variable_scope('conv_6'):
                net = self._conv_layer(net, 128, 1, 1,'conv_6')
            with tf.variable_scope('conv_7'):
                net = self._conv_layer(net, 256, 3, 1,'conv_7')
            with tf.variable_scope('conv_8'):
                net = self._conv_layer(net, 256, 1, 1,'conv_8')
            with tf.variable_scope('conv_9'):
                net = self._conv_layer(net, 512, 3, 1,'conv_9')
            net = self._maxpool_layer(net, 2, 2)
            with tf.variable_scope('conv_11'):
                net = self._conv_layer(net, 256, 1, 1,'conv_11')
            with tf.variable_scope('conv_12'):
                net = self._conv_layer(net, 512, 3, 1,'conv_12')
            with tf.variable_scope('conv_13'):
                net = self._conv_layer(net, 256, 1, 1,'conv_13')
            with tf.variable_scope('conv_14'):
                net = self._conv_layer(net, 512, 3, 1,'conv_14')
            with tf.variable_scope('conv_15'):
                net = self._conv_layer(net, 256, 1, 1,'conv_15')
            with tf.variable_scope('conv_16'):
                net = self._conv_layer(net, 512, 3, 1,'conv_16')
            with tf.variable_scope('conv_17'):
                net = self._conv_layer(net, 256, 1, 1,'conv_17')
            with tf.variable_scope('conv_18'):
                net = self._conv_layer(net, 512, 3, 1,'conv_18')
            with tf.variable_scope('conv_19'):
                net = self._conv_layer(net, 512, 1, 1,'conv_19')
            with tf.variable_scope('conv_20'):
                net = self._conv_layer(net, 1024, 3, 1,'conv_20')
            net = self._maxpool_layer(net, 2, 2)
            with tf.variable_scope('conv_22'):
                net = self._conv_layer(net,  512, 1, 1,'conv_22')
            with tf.variable_scope('conv_23'):
                net = self._conv_layer(net,  1024, 3, 1,'conv_23')
            with tf.variable_scope('conv_24'):
                net = self._conv_layer(net,  512, 1, 1,'conv_24')
            with tf.variable_scope('conv_25'):
                net = self._conv_layer(net,  1024, 3, 1,'conv_25')
            with tf.variable_scope('conv_26'):
                net = self._conv_layer(net,  1024, 3, 1,'conv_26')
            with tf.variable_scope('conv_28'):
                net = self._conv_layer(net,  1024, 3, 2,'conv_28')
            with tf.variable_scope('conv_29'):
                net = self._conv_layer(net,  1024, 3, 1,'conv_29')
            with tf.variable_scope('conv_30'):
                net = self._conv_layer(net,  1024, 3, 1,'conv_30')
            net = self._flatten(net)
            with tf.variable_scope('fc_33'):
                net = self._fc_layer(net,  512, activation=self.leak_relu,scope='fc_33')
            with tf.variable_scope('fc_34'):
                net = self._fc_layer(net, 4096, activation=self.leak_relu,scope='fc_34')
            with tf.variable_scope('fc_36'):
                net = self._fc_layer(net, 7*7*30,scope='fc_36')
        return net,x

    def _conv_layer(self, x, num_filters, filter_size, stride,scope):

        in_channels = x.get_shape().as_list()[-1]
        weight = tf.Variable(tf.truncated_normal([filter_size, filter_size,
                                                  in_channels, num_filters], stddev=0.1),name='weights')
        bias = tf.Variable(tf.zeros([num_filters,]),name='biases')

        pad_size = filter_size // 2
        pad_mat = np.array([[0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
        x_pad = tf.pad(x, pad_mat)
        conv = tf.nn.conv2d(x_pad, weight, strides=[1, stride, stride, 1], padding="VALID",name=scope)
        output = self.leak_relu(tf.nn.bias_add(conv, bias))

        return output

    def _fc_layer(self, x,  num_out, activation=None,scope=None):

        num_in = x.get_shape().as_list()[-1]
        weight = tf.Variable(tf.truncated_normal([num_in, num_out], stddev=0.1),name='weights')
        bias = tf.Variable(tf.zeros([num_out,]),name='biases')
        output = tf.nn.xw_plus_b(x, weight, bias,name=scope)
        if activation:
            output = activation(output)

        return output

    def _maxpool_layer(self, x,  pool_size, stride):
        output = tf.nn.max_pool(x, [1, pool_size, pool_size, 1],
                                strides=[1, stride, stride, 1], padding="SAME")

        return output

    def _flatten(self, x):
        """flatten the x"""
        tran_x = tf.transpose(x, [0, 3, 1, 2])  # channle first mode
        nums = np.product(x.get_shape().as_list()[1:])
        return tf.reshape(tran_x, [-1, nums])


#############   IOU

    def filter(self,predicition):
        cls = tf.reshape(predicition[0,:7*7*20],[7,7,20])
        confidence = tf.reshape(predicition[0,7*7*20:7*7*20 + 7*7*2],[7,7,2])
        boxes = tf.reshape(predicition[0,7*7*20 + 7*7*2:],[7,7,2,4])

        #true box = (x,y,w**2,h**2) 乘以图像的宽度和高度
        boxes = tf.stack(
            [
                (boxes[:,:,:,0] + tf.constant(self.x_offset,dtype=tf.float32)) / 7 * self.img_shape[0],
                (boxes[:,:,:,1] + tf.constant(self.y_offset,dtype=tf.float32)) / 7 * self.img_shape[1],
                tf.square(boxes[:,:,:,2]) * self.img_shape[0],
                tf.square(boxes[:,:,:,3]) * self.img_shape[1]
             ],axis=3
        )
        scores = tf.expand_dims(confidence, -1) * tf.expand_dims(cls, 2)
        print(scores)
        scores = tf.reshape(scores, [-1, 20])

        boxes = tf.reshape(boxes, [-1, 4])

        #拿到每个box的类别与得分
        box_classes = tf.argmax(scores, axis=1)
        box_class_scores = tf.reduce_max(scores, axis=1)

        #过滤
        filter_mask = box_class_scores >= self.threshold
        scores = tf.boolean_mask(box_class_scores, filter_mask)
        boxes = tf.boolean_mask(boxes, filter_mask)
        box_classes = tf.boolean_mask(box_classes, filter_mask)

        _boxes = tf.stack([boxes[:, 0] - 0.5 * boxes[:, 2], boxes[:, 1] - 0.5 * boxes[:, 3],
                           boxes[:, 0] + 0.5 * boxes[:, 2], boxes[:, 1] + 0.5 * boxes[:, 3]], axis=1)

        nms_indices = tf.image.non_max_suppression(_boxes, scores,
                                                   self.max_output_size, self.iou_threshold)
        scores = tf.gather(scores, nms_indices)
        boxes = tf.gather(boxes, nms_indices)
        box_classes = tf.gather(box_classes, nms_indices)

        return scores,boxes,box_classes

    def _detect_from_image(self, image):
        """Do detection given a cv image"""


        img_resized = cv2.resize(image, (448, 448))
        self.img = img_resized
        img_RGB = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_RGB = np.expand_dims(img_RGB,0)


        img_resized_np = np.asarray(img_RGB)
        _images = np.zeros((1, 448, 448, 3), dtype=np.float32)
        _images[0] = (img_resized_np / 255.0) * 2.0 - 1.0


        return _images

    def draw_rectangle(self,img, classes, scores, bboxes, colors, thickness=2):

        for i in range(bboxes.shape[0]):
            x = int(bboxes[i][0])
            y = int(bboxes[i][1])
            w = int(bboxes[i][2]) // 2
            h = int(bboxes[i][3]) // 2

            print("[x, y, w, h]=[%d, %d, %d, %d]" % (x, y, w, h))


            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), colors[0], thickness)
            # Draw text...
            s = '%s/%.3f' % (self.classes[classes[i]], scores[i])
            cv2.rectangle(img, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
            cv2.putText(img, s, (x - w + 5, y - h - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.namedWindow("img", 0);
        cv2.resizeWindow("img", 640, 480);
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def calc_iou(self,bboxes1, bboxes2):
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


    def loss_layer(self, predicts, labels, scope='loss_layer'):


        #label为（(45,7,7,25)）  5个为盒子信息  (x,y,w,h,c)  后20个为类别
        with tf.variable_scope(scope):
            # 预测值
            # class-20
            predict_classes = tf.reshape(
                predicts[:, :7 * 7 * 20],
                [self.batch_size, 7, 7, 20])
            # confidence-2
            predict_confidence = tf.reshape(
                predicts[:, 7 * 7 * 20:7 * 7 * 20 + 7 * 7 * 2],
                [self.batch_size, 7, 7, 2])
            # bounding box-2*4
            predict_boxes = tf.reshape(
                predicts[:, 7 * 7 * 20 + 7 * 7 * 2:],
                [self.batch_size, 7, 7, 2, 4])

            # 实际值
            # shape(45,7,7,1)
            # response中的值为0或者1.对应的网格中存在目标为1，不存在目标为0.
            # 存在目标指的是存在目标的中心点，并不是说存在目标的一部分。所以，目标的中心点所在的cell其对应的值才为1，其余的值均为0
            response = tf.reshape(
                labels[..., 0],
                [self.batch_size, 7, 7, 1])
            # shape(45,7,7,1,4)
            boxes = tf.reshape(
                labels[..., 1:5],
                [self.batch_size, 7, 7, 1, 4])
            # shape(45,7,7,2,4),boxes的四个值，取值范围为0~1
            boxes = tf.tile(
                boxes, [1, 1, 1, 2, 1]) / self.img_shape[0]
            # shape(45,7,7,20)
            classes = labels[..., 5:]

            # self.offset shape(7,7,2)
            # offset shape(1,7,7,2)

            # shape(45,7,7,2)
            x_offset = tf.tile(self.x_offset, [self.batch_size, 1, 1, 1])  #(45,7,7,2)
            # shape(45,7,7,2)
            y_offset = tf.transpose(x_offset, (0, 2, 1, 3))

            # convert the x, y to the coordinates relative to the top left point of the image
            # the predictions of w, h are the square root
            # shape(45,7,7,2,4)  ->(x,y,w,h)
            predict_boxes_tran = tf.stack(
                [(predict_boxes[..., 0] + x_offset) / 7,
                 (predict_boxes[..., 1] + y_offset) / 7,
                 tf.square(predict_boxes[..., 2]),
                 tf.square(predict_boxes[..., 3])], axis=-1)

            # 预测box与真实box的IOU,shape(45,7,7,2)
            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

            # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            # shape(45,7,7,1), find the maximum iou_predict_truth in every cell
            # 在训练时，如果该单元格内确实存在目标，那么只选择IOU最大的那个边界框来负责预测该目标，而其它边界框认为不存在目标
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
            # object probs (45,7,7,2)
            object_mask = tf.cast(
                (iou_predict_truth >= object_mask), tf.float32) * response

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            # noobject confidence(45,7,7,2)
            noobject_probs = tf.ones_like(
                object_mask, dtype=tf.float32) - object_mask

            # shape(45,7,7,2,4)，对boxes的四个值进行规整，xy为相对于网格左上角，wh为取根号后的值，范围0~1
            boxes_tran = tf.stack(
                [boxes[..., 0] * 7 - x_offset,
                 boxes[..., 1] * 7 - y_offset,
                 tf.sqrt(boxes[..., 2]),
                 tf.sqrt(boxes[..., 3])], axis=-1)

            # class_loss shape(45,7,7,20)
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                name='class_loss') * self.class_scale

            # object_loss  confidence=iou*p(object)
            # p(object)的值为1或0
            object_delta = object_mask * (predict_confidence - iou_predict_truth)
            object_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                name='object_loss') * self.object_scale

            # noobject_loss  p(object)的值为0
            noobject_delta = noobject_probs * predict_confidence
            noobject_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                name='noobject_loss') * self.noobject_scale

            # coord_loss
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                name='coord_loss') * self.coord_scale

            return class_loss+object_loss+noobject_loss+coord_loss


    def train_yolo(self):
        global_step = tf.train.create_global_step()
        learning_rate = tf.train.exponential_decay(
            0.0001, global_step, 30000,
            0.1, True, name='learning_rate')
        op = tf.train.GradientDescentOptimizer(learning_rate).minimize()








if __name__ == '__main__':
    # import numpy as np
    # from tensorflow.python import pywrap_tensorflow
    #
    # # checkpoint_path = '../Nn/ssd_vgg_300_weights.ckpt'
    # # checkpoint_path = '../NN/ssd_vgg_300_weights.ckpt'
    # checkpoint_path = './model/faster_rcnn/VGGnet_fast_rcnn_iter_70000.ckpt'
    # # print(path.getcwdu())
    # # print(checkpoint_path)
    # # read data from checkpoint file
    # reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
    # var_to_shape_map = reader.get_variable_to_shape_map()
    #
    # data_print = np.array([])
    # for key in var_to_shape_map:
    #     print('tensor_name', key)
    #     ckpt_data = np.array(reader.get_tensor(key))  # cast list to np arrary
    #     print(ckpt_data.shape)
    #     ckpt_data = ckpt_data.flatten()  # flatten list
    #     data_print = np.append(data_print, ckpt_data, axis=0)
    #
    # # print(data_print, data_print.shape, np.max(data_print), np.min(data_print), np.mean(data_print))
    # print(data_print)



    yo = Yolo()

    pred,x = yo._build_net()
    init = tf.global_variables_initializer()
    sess = tf.Session()
    checkpoint_path = '../../model/yolo/YOLO_small.ckpt'
    saver = tf.train.Saver()
    sess.run(init)
    saver.restore(sess, checkpoint_path)

    img = cv2.imread('../../road.jpg')
    img = yo._detect_from_image(img)

    scores, boxes, box_classes = yo.filter(pred)
    scores, boxes, box_classes = sess.run([scores, boxes, box_classes],feed_dict={x:img})
    print(scores, boxes, box_classes)

    yo.draw_rectangle(yo.img,box_classes,scores,boxes,[[0,0,255],[255,0,0]])







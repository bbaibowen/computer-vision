import tensorflow as tf
from anchor import gen_anchors
import numpy as np
import cv2



class Siam_RPN:

    def __init__(self):
         self.target_holder = tf.placeholder(dtype=tf.float32,shape=[1,127,127,3])
         self.search_holder = tf.placeholder(dtype=tf.float32,shape=[1,255,255,3])
         self.k = 5
         self.anchors = tf.convert_to_tensor(gen_anchors())

    def get_var(self,name, shape, initializer, weightDecay = 0.0, dType=tf.float32, trainable = True):


        if weightDecay > 0:
            regularizer = tf.contrib.layers.l2_regularizer(weightDecay)
        else:
            regularizer = None

        return tf.get_variable(name, shape=shape, initializer=initializer, dtype=dType, regularizer=regularizer, trainable=trainable)


    def conv2d(self,x,filters,k_size,stride,iter = 1,padding = 'VALID',name = 'conv2d'):

        in_channel = int(x.get_shape()[-1])
        conv = lambda i,k:tf.nn.conv2d(i,k,[1,stride,stride,1],padding=padding)
        with tf.variable_scope(name) as scope:

            w = self.get_var('weights', shape=[k_size, k_size, in_channel / iter, filters],
                                 initializer=tf.truncated_normal_initializer(stddev=0.03), weightDecay=5e-04,
                                 dType=tf.float32, trainable=True)
            b = self.get_var('biases', shape=[filters, ],
                                 initializer=tf.constant_initializer(value=0.1, dtype=tf.float32), weightDecay=0,
                                 dType=tf.float32, trainable=True)

            if iter == 1:
                net = conv(x,w)
            else:
                x_group = tf.split(x,iter,3)
                w_group = tf.split(w,iter,3)
                net = [conv(i,k) for i,k in zip(x_group,w_group)]
                net = tf.concat(net,3)

            net = tf.nn.bias_add(net,b)
            net = tf.nn.relu(net)

            return net

    def lrn(self,x,radius, alpha, beta,bias = 1.0):

        return tf.nn.local_response_normalization(x,depth_radius=radius,alpha = alpha,beta=beta,bias=2.0,name='lrn')


    def bulid_alexnet(self,x):

        with tf.variable_scope('conv1'):

            net = self.conv2d(x,96,11,2)
            net = self.lrn(net,2,1e-4,0.75)
            net = tf.nn.max_pool(net,[1,3,3,1],[1,2,2,1],padding='VALID',name='pool1')

        with tf.variable_scope('conv2'):

            net = self.conv2d(net,256,5,1,iter=2)
            net = self.lrn(net,2,1e-4,0.75)
            net = tf.nn.max_pool(net,[1,3,3,1],[1,2,2,1],padding='VALID',name='pool2')

        with tf.variable_scope('conv3'):

            net = self.conv2d(net,384,3,1)

        with tf.variable_scope('conv4'):
            net = self.conv2d(net,384,3,1,iter=2)

        with tf.variable_scope('conv5'):
            net = self.conv2d(net,256,3,1,iter=2)

        return net





    def build_train(self,target,search):

        #ALEXNET
        with tf.variable_scope('alex') as scope:
            target = self.bulid_alexnet(target)
            print(target)
            scope.reuse_variables()
            search = self.bulid_alexnet(search)

        with tf.variable_scope('RPN_target'):

            target_net = self.conv2d(target,2 * self.k * 256,3,1,name='t_cls')
            target_net = tf.reshape(target_net,(tf.shape(target_net)[1],tf.shape(target_net)[2],256,10))
            target_box = self.conv2d(target,4 * self.k * 256,3,1,name='t_box')
            target_box = tf.reshape(target_box,(tf.shape(target_box)[1],tf.shape(target_box)[2],256,20))

        with tf.variable_scope('RPN_search'):

            search1 = self.conv2d(search,256,3,1,name='s_cls')
            search2 = self.conv2d(search,256,3,1,name='s_box')
            print(search1,search2)

        with tf.variable_scope('merge'):

            cls = tf.nn.conv2d(search1,target_net,strides=[1,1,1,1],padding='VALID',name='cls')  #[1,17,17,2k]
            reg = tf.nn.conv2d(search2,target_box,strides=[1,1,1,1],padding='VALID',name='reg')  #[1,17,17,4k]

        return cls,reg

    def center_to_corner(self,box):
        t_1=box[:,0]-(box[:,2]-1)/2
        t_2=box[:,1]-(box[:,3]-1)/2
        t_3=box[:,0]+(box[:,2]-1)/2
        t_4=box[:,1]+(box[:,3]-1)/2
        box_temp=tf.transpose(tf.stack([t_1,t_2,t_3,t_4],axis=0),(1,0))
        return box_temp
    def corner_to_center(self,box):
        t_1=box[:,0]+(box[:,2]-box[:,0])/2
        t_2=box[:,1]+(box[:,3]-box[:,1])/2
        t_3=(box[:,2]-box[:,0])
        t_4=(box[:,3]-box[:,1])
        box_temp=tf.transpose(tf.stack([t_1,t_2,t_3,t_4],axis=0),(1,0))
        return box_temp


    def diff_anchor_gt(self,gt,anchors):
        #gt [x,y,w,h]
        #anchors [x,y,w,h]
        t_1=(gt[0]-anchors[:,0])/(anchors[:,2]+0.01)
        t_2=(gt[1]-anchors[:,1])/(anchors[:,3]+0.01)
        t_3=tf.log(gt[2]/(anchors[:,2]+0.01))
        t_4=tf.log(gt[3]/(anchors[:,3]+0.01))
        diff_anchors=tf.transpose(tf.stack([t_1,t_2,t_3,t_4],axis=0),(1,0))
        return diff_anchors#[dx,dy,dw,dh]
    def iou(self,box1,box2):
        """ Intersection over Union (iou)
            Args:
                box1 : [N,4]
                box2 : [K,4]
                box_type:[x1,y1,x2,y2]
            Returns:
                iou:[N,K]
        """
        N=box1.get_shape()[0]
        K=box2.get_shape()[0]
        box1=tf.reshape(box1,(N,1,4))+tf.zeros((1,K,4))#box1=[N,K,4]
        box2=tf.reshape(box2,(1,K,4))+tf.zeros((N,1,4))#box1=[N,K,4]
        x_max=tf.reduce_max(tf.stack((box1[:,:,0],box2[:,:,0]),axis=-1),axis=2)
        x_min=tf.reduce_min(tf.stack((box1[:,:,2],box2[:,:,2]),axis=-1),axis=2)
        y_max=tf.reduce_max(tf.stack((box1[:,:,1],box2[:,:,1]),axis=-1),axis=2)
        y_min=tf.reduce_min(tf.stack((box1[:,:,3],box2[:,:,3]),axis=-1),axis=2)
        tb=x_min-x_max
        lr=y_min-y_max
        zeros=tf.zeros_like(tb)
        tb=tf.where(tf.less(tb,0),zeros,tb)
        lr=tf.where(tf.less(lr,0),zeros,lr)
        over_square=tb*lr
        all_square=(box1[:,:,2]-box1[:,:,0])*(box1[:,:,3]-box1[:,:,1])+(box2[:,:,2]-box2[:,:,0])*(box2[:,:,3]-box2[:,:,1])-over_square
        return over_square/all_square


    def anchor_target_layer(self,gt,anchors):

        # gt [x,y,w,h]
        # anchors [x1,y1,x2,y2]

        zeros = tf.zeros_like(anchors)
        ones = tf.ones_like(anchors)

        #clip
        all_box = tf.where(tf.less(anchors, 0), zeros, anchors)
        all_box = tf.where(tf.greater(all_box, 255), ones * 255, all_box)


        target_inside_weight_box = tf.zeros((anchors.get_shape()[0], 4), dtype=tf.float32)
        target_outside_weight_box = tf.ones((anchors.get_shape()[0], 4), dtype=tf.float32)
        label = -tf.ones((anchors.get_shape()[0],), dtype=tf.float32)

        gt_array = tf.reshape(gt, (1, 4))
        gt_array = self.center_to_corner(gt_array)

        iou_value = tf.reshape(self.iou(all_box, gt_array), [-1])

        pos_value, pos_index = tf.nn.top_k(iou_value, 16)
        pos_mask_label = tf.ones_like(label)
        label = tf.where(tf.greater_equal(iou_value, pos_value[-1]), pos_mask_label, label)

        neg_index = tf.reshape(tf.where(tf.less(iou_value, 0.3)), [-1])
        neg_index = tf.random_shuffle(neg_index)
        neg_index = neg_index[0:16 * 3]
        neg_index = tf.reduce_sum(tf.one_hot(neg_index, anchors.get_shape()[0]), axis=0)
        neg_mask_label = tf.zeros_like(label)
        label = tf.where(tf.equal(neg_index, 1), neg_mask_label, label)

        target_box = self.diff_anchor_gt(gt, self.corner_to_center(all_box))
        temp_label = tf.transpose(tf.stack([label, label, label, label], axis=0), (1, 0))
        target_inside_weight_box = tf.where(tf.equal(temp_label, 1), temp_label, target_inside_weight_box)
        target_outside_weight_box = target_outside_weight_box * 1.0 / 48.
        return label, target_box, target_inside_weight_box, target_outside_weight_box, all_box



    def loss(self,gt,p_cls,p_box):

        label, target_box, target_inside_weight, target_outside_weight, all_box = \
                        self.anchor_target_layer(gt,self.anchors)

        #cls
        p_cls = tf.reshape(p_cls,(-1,2))
        p_cls_mask = tf.where(tf.not_equal(label,-1))
        p_cls = tf.gather(p_cls,p_cls_mask)
        p_cls = tf.reshape(p_cls,(-1,2))
        p_cls_label = tf.cast(
            tf.reshape(tf.gather(label,tf.where(tf.not_equal(label,-1))),[-1]),tf.int32
        )
        cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=p_cls,labels=p_cls_label)
        cls_loss = tf.reduce_mean(cls_loss)

        p_box = tf.reshape(p_box,(-1,4))
        # print(p_box,p_box.shape)
        # print(target_box.shape)
        index = tf.multiply(target_inside_weight,(p_box-target_box))
        p_box_mask = tf.cast(tf.less(tf.abs(index),1),tf.float32)
        # lr1 = index * index * 0.5
        lr1 = tf.multiply(tf.multiply(index,index),0.5)
        # lr2 = tf.abs(index) - 0.5
        lr2 = tf.subtract(tf.abs(index),0.5)
        box_loss = tf.multiply(tf.add(tf.multiply(lr1,p_box_mask),tf.multiply(lr2,(1. - p_box_mask))),target_outside_weight)
        box_loss = tf.reduce_sum(box_loss)
        loss_total = tf.add(cls_loss,box_loss)

        return loss_total,[p_box,target_box]




if __name__ == '__main__':
    test = Siam_RPN()

    cls, reg = test.build_train()
    print(cls,reg)
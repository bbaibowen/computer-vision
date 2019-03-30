import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from lanenet_discriminative_loss import discriminative_loss


'''

lanenet的基础model可以选择很多，这里我们选择我们比较熟悉的vgg16

'''


class Lanenet(object):
    def __init__(self,which):

        self.is_train = True if which is 'train' else False

        self.VGG_MEAN = [103.939, 116.779, 123.68]
        self.img_size = (256,512)
        self._color_map = [np.array([255, 0, 0]),
                           np.array([0, 255, 0]),
                           np.array([0, 0, 255]),
                           np.array([125, 125, 0]),
                           np.array([0, 125, 125]),
                           np.array([125, 0, 125]),
                           np.array([50, 100, 50]),
                           np.array([100, 50, 100])]


    def conv2d(self,x,num_filters,k_size,padding='SAME',stride=1,use_bias = False,name = 'conv',w_name = 'W'):

        with tf.variable_scope(name):
            input_shape = x.get_shape().as_list()
            in_channel = input_shape[3]
            filter_shape = [k_size,k_size] + [in_channel,num_filters]
            strides = [1,stride,stride,1]
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer()

            w = tf.get_variable(w_name,filter_shape,initializer=w_init)

            if use_bias:
                b = tf.get_variable('b', [num_filters], initializer=b_init)

            conv = tf.nn.conv2d(x,w,strides,padding)

            ret = tf.identity(tf.nn.bias_add(conv,b) if use_bias else conv,name = name)


            return ret

    def conv2d_block(self,x,num_filters,k_size,name,stride = 1,padding = 'SAME'):

        with tf.variable_scope(name):

            #conv
            net = self.conv2d(x,num_filters,k_size,stride=stride,padding=padding,name = 'conv')

            #bn
            net = tf.layers.batch_normalization(inputs=net,training=self.is_train,name = 'bn')

            #relu
            out = tf.nn.relu(net,name='relu')

            return out

    def maxpooling(self,x,k_size,stride = None,padding='VALID',name='maxpool'):

        k_size = [1,k_size,k_size,1]

        stride = k_size if stride is None else [1,stride,stride,1]

        return tf.nn.max_pool(x,k_size,stride,padding,name=name)


    #反卷积
    def deconv2d(self,x,filters,k_size,padding = 'SAME',stride = 1,use_bias = False,activation = None,trainable = True,name = None):

        with tf.variable_scope(name):

            input_shape = x.get_shape().as_list()
            w_init = tf.contrib.layers.variance_scaling_initializer()
            b_init = tf.constant_initializer()

            out = tf.layers.conv2d_transpose(x,filters=filters,kernel_size=k_size,strides=stride,padding=padding,activation=activation,
                                             use_bias=use_bias,kernel_initializer=w_init,bias_initializer=b_init,trainable=trainable,
                                             name = name)
            return out


    def build_vgg16net(self,x,name):

        with tf.variable_scope(name):

            #s1
            conv1_1 = self.conv2d_block(x,64,3,'conv1_1')
            conv1_2 = self.conv2d_block(conv1_1,64,3,'conv1_2')
            pool1 = self.maxpooling(conv1_2,2,2,name='pool1')

            #s2
            conv2_1 = self.conv2d_block(pool1,128,3,'conv2_1')
            conv2_2 = self.conv2d_block(conv2_1,128,3,'conv2_2')
            pool2 = self.maxpooling(conv2_2,2,2,name='pool2')

            #s3
            conv3_1 = self.conv2d_block(pool2,256,3,'conv3_1')
            conv3_2 = self.conv2d_block(conv3_1,256,3,'conv3_2')
            conv3_3 = self.conv2d_block(conv3_2,256,3,'conv3_3')
            pool3 = self.maxpooling(conv3_3,2,2,name='pool3')

            #s4
            conv4_1 = self.conv2d_block(pool3,512,3,'conv4_1')
            conv4_2 = self.conv2d_block(conv4_1,512,3,'conv4_2')
            conv4_3 = self.conv2d_block(conv4_2,512,3,'conv4_3')
            pool4 = self.maxpooling(conv4_3,2,2,name='pool4')

            #s5
            conv5_1 = self.conv2d_block(pool4, 512, 3, 'conv5_1')
            conv5_2 = self.conv2d_block(conv5_1, 512, 3, 'conv5_2')
            conv5_3 = self.conv2d_block(conv5_2, 512, 3, 'conv5_3')
            pool5 = self.maxpooling(conv5_3, 2, 2, name='pool5')

            #pool3-5都是后续需要的feature map



            return [pool5,pool4,pool3]   #解码网络是由深到浅解码

    def decode_layer(self,input,name):
        #解码特征信息反卷积还原

        with tf.variable_scope(name):

            # score stage 1
            score = self.conv2d(input[0],64,1,name='score_origin')

            decode_layer_list = input[1:]
            for i in range(len(decode_layer_list)):

                conv_tranpose = self.deconv2d(score,64,k_size=4,stride=2,name = 'deconv_{:d}'.format(i + 1))

                score = self.conv2d(decode_layer_list[i],64,1,name='score_{:d}'.format(i + 1))

                fused = tf.add(conv_tranpose,score,name='fuse_{:d}'.format(i + 1))
                score = fused


            deconv_final = self.deconv2d(score,64,k_size=16,stride=8,name='deconv_final')
            score_final = self.conv2d(deconv_final,2,1,name='score_final')  #logits

            return deconv_final,score_final


    def build_model(self,x):


    #1 、 forward获取logits


        feats = self.build_vgg16net(x,'encode')

        #decode部分，decode部分是用FCN网络（全卷积）
        deconv_final, score_final = self.decode_layer(feats,'decode')
        # print(deconv_final,score_final)

        return deconv_final, score_final

    def logits(self,deconv_final, score_final):

        binary_seg_ret = tf.nn.softmax(score_final)
        binary_seg_ret = tf.argmax(binary_seg_ret, axis=-1)

        # 像素嵌入
        embedding = self.conv2d(deconv_final, 4, 1, name='pix_embedding_conv')
        embedding = tf.nn.relu(embedding, name='pix_embedding_relu')


        return binary_seg_ret,embedding



    #然后这里是一些图像的操作

    #，闭运算 + 连通域分析
    def process(self,img,kernel_size = 5,minarea_threshold=15):

        #
        img_cp = np.array(img,np.uint8)
        # img_cp = cv2.cvtColor(img_cp, cv2.COLOR_BGR2GRAY)

        #自定义一个核
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(kernel_size, kernel_size))
        #闭运算
        morphological_ret = cv2.morphologyEx(img_cp, cv2.MORPH_CLOSE, kernel, iterations=1)
        # morphological_ret_cp = morphological_ret
        #连通域分析
        img_cp_ = cv2.connectedComponentsWithStats(morphological_ret, connectivity=8, ltype=cv2.CV_32S)

        #排序并删除过小的连通域
        labels = img_cp_[1]
        stats = img_cp_[2]

        for index, stat in enumerate(stats):
            if stat[4] <= minarea_threshold:
                idx = np.where(labels == index)
                morphological_ret[idx] = 0

        return morphological_ret



    ######################实现LaneNet中实例分割的聚类部分###############
    def cluster(self,binary_seg_ret,instance_seg_ret,im,bandwidth = 1.5):

        #step 1:通过二值分割掩码图在实例分割图上获取所有车道线的特征向量
        id = np.where(binary_seg_ret == 1)
        lane_embedding_feats = []
        lane_coordinate = []

        for i in range(len(id[0])):
            lane_embedding_feats.append(instance_seg_ret[id[0][i],id[1][i]])
            lane_coordinate.append([id[0][i],id[1][i]])
        lane_embedding_feats = np.array(lane_embedding_feats,np.float32)
        lane_coordinate = np.array(lane_coordinate,np.uint64)

        print(lane_embedding_feats.shape,lane_coordinate.shape)

        #step 2:mean shift聚类
        meanshift = MeanShift(bandwidth,bin_seeding=True)
        meanshift.fit(lane_embedding_feats)
        labels = meanshift.labels_            #类别
        centers = meanshift.cluster_centers_  #中心点
        num_cls = centers.shape[0]            #多少类

        # print(num_cls)
        # mask = np.zeros(shape = [binary_seg_ret.shape[0],binary_seg_ret.shape[1],3],dtype = np.uint8)
        for index,i in enumerate(range(num_cls)):
            id_ = np.where(labels == i)
            coord = lane_coordinate[id_]
            coord = np.flip(coord,axis=1)
            coord = np.array([coord])
            color = (int(self._color_map[index][0]),
                     int(self._color_map[index][1]),
                     int(self._color_map[index][2]))

            # print(coord.shape)
            # print(coord)
            cv2.polylines(img=im, pts=coord, isClosed=False, color=color, thickness=2)


    def loss(self, binary_label, instance_label,logits,deconv_final, name):

        with tf.variable_scope(name):

            #计算二值分割损失
            binary_label_shape = binary_label.get_shape().as_list()
            binary_label_ = tf.reshape(binary_label,[binary_label_shape[0] * binary_label_shape[1] * binary_label_shape[2]])
                #加入class weights
            # unique_labels, unique_id, counts = tf.unique_with_counts(binary_label_)
            # counts = tf.cast(counts,tf.float32)
            # weights = tf.divide(1.,tf.log(tf.add(tf.divide(tf.constant(1.),counts),tf.constant(1.02))))
            # weights = tf.gather(weights,binary_label)
            binary_losses= tf.losses.sparse_softmax_cross_entropy(labels=binary_label,logits=logits)
            binary_losses = tf.reduce_mean(binary_losses)


            #discriminative loss
            embedding = self.conv2d(deconv_final, 4, 1, name='pix_embedding_conv')
            embedding = tf.nn.relu(embedding, name='pix_embedding_relu')
            img_shape = embedding.get_shape().as_list()
            disc_loss,l_var,l_dist,l_reg = \
                discriminative_loss(embedding,instance_label,4,(img_shape[1],img_shape[2]),0.5,3.0,1.0,1.0,1e-3)

            l2_loss = tf.constant(.0,tf.float32)
            for i in tf.trainable_variables():
                if 'bn' in i.name:
                    continue
                else:
                    l2_loss = tf.add(l2_loss,tf.nn.l2_loss(i))


            all_losses = .5 * binary_losses + .5 * disc_loss + l2_loss

            return all_losses


if __name__ == '__main__':


    x = tf.placeholder(dtype=tf.float32,shape=[None,256,512,3])
    te = Lanenet('test')

    # img_path = '../0002.png'
    img_path = '../0002.png'
    ckpt_path = '../model/lanenet/tusimple_lanenet.ckpt'
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image_vis = image
    image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
    img_ = image
    image = image - te.VGG_MEAN
    image = np.expand_dims(image,0)
    with tf.variable_scope('lanenet_model'):

        with tf.variable_scope('inference'):

            deconv_final, score_final = te.build_model(x)
        binary_seg_ret, embedding = te.logits(deconv_final, score_final)
        print(binary_seg_ret[0])

    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess,ckpt_path)

    binary_seg_ret, embedding = sess.run([binary_seg_ret, embedding],feed_dict={x:image})
    print(binary_seg_ret.shape,embedding.shape)


    # binary_seg_ret[0] = te.process(binary_seg_ret[0]) #闭运算这一步没有必要，差别不大其实

    te.cluster(binary_seg_ret[0],embedding[0],img_)

    plt.figure('pred')
    plt.imshow(binary_seg_ret[0])
    plt.figure('embedding')
    plt.imshow(embedding[0])
    plt.figure('src')
    plt.imshow(img_[:,:,(2,1,0)])
    # plt.imshow(mask[0])
    plt.show()


    print('done')
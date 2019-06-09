'''
DSOD:其实就是SSD + DenseNet，一个不追求速度和精度的目标检测
    唯一惊喜的是可以从零训练（其实SSD也可以从零训练，只是精度下降）
    因为之前从头实现过SSD，而DSOD其实只是换了网络，大多数还是沿用SSD，训练也是，这里就写densenet吧
'''

import tensorflow.layers as tl
import tensorflow as tf


def dense_block(x,iter,two_conv,one_conv,is_train = False,name = 'denseblock'):

    with tf.variable_scope(name):
        net = x
        for i in range(iter):
            x = tl.batch_normalization(net,trainable=is_train,name = name + '_bn1/' + str(i))
            x = tf.nn.relu(x)
            x = tl.conv2d(x,one_conv,(1,1),padding='same',name=name + '_conv1/' + str(i))
            x = tl.batch_normalization(x,trainable=is_train,name = name + '_bn2/' + str(i))
            x = tf.nn.relu(x)
            x = tl.conv2d(x,two_conv,(3,3),(1,1),padding='same',name = name + '_conv2/' + str(i))
            net = tf.concat([x,net],-1)
        return net
def denseblock_pl(x,step,channel,is_train = True,name = 'denseblock_pl',padding = 'same'):

    with tf.variable_scope(name):
        net = tl.max_pooling2d(x,(2,2),(2,2),padding=padding)
        net = tl.batch_normalization(net,trainable=is_train,name = name + '_bn1')
        net = tf.nn.relu(net)
        net = tl.conv2d(net,channel,(1,1),(1,1),padding='same',name=name + '_conv1')

        net2 = tl.batch_normalization(x, trainable=is_train, name=name + '_bn2')
        net2 = tf.nn.relu(net2)
        net2 = tl.conv2d(net2,channel,(1,1),(1,1),padding='same',name=name + '_conv2')
        net2 = tl.batch_normalization(net2, trainable=is_train, name=name + '_bn3')
        net2 = tf.nn.relu(net2)
        net2 = tl.conv2d(net2,step,(3,3),(2,2),padding=padding,name=name + '_conv4')

        out = tf.concat([net,net2],axis=-1)

        return out



def main_net(x,is_train):



    with tf.variable_scope('DSOD'):

        #STEM
        net = tl.conv2d(x,64,(3,3),(2,2),padding='same',name='stem1')
        net = tl.batch_normalization(net,trainable=is_train,name='stem1_bn')
        net = tf.nn.relu(net)
        net = tl.conv2d(net,64,(3,3),(1,1),padding='same',name='stem2')
        net = tl.batch_normalization(net, trainable=is_train,name='stem2_bn')
        net = tf.nn.relu(net)
        net = tl.conv2d(net, 128, (3, 3), (1, 1), padding='same', name='stem3')
        net = tl.batch_normalization(net, trainable=is_train, name='stem3_bn')
        net = tf.nn.relu(net)
        net = tl.max_pooling2d(net,(2,2),(2,2),name='stem3_pool3')

        #DenseBlock 1  6x
        net = dense_block(net,iter=6,two_conv=48,one_conv=192,is_train = is_train,name = 'denseblock1')


        #Transition Layer 1
        net = tl.batch_normalization(net,trainable=is_train,name='transition1_bn')
        net = tf.nn.relu(net)
        net = tl.conv2d(net,416,(1,1),(1,1),padding='same',name='transition1_conv1')
        net = tl.max_pooling2d(net,(2,2),(2,2),padding='same',name='transition1_pool1')

        #DenseBlock 2   8x
        net = dense_block(net,8,48,192,is_train = False,name='denseblock2')

        #Transition Layer 2
        net = tl.batch_normalization(net,trainable=is_train,name='transition2_bn')
        net = tf.nn.relu(net)
        net = tl.conv2d(net,800,(1,1),(1,1),padding='same',name='transition2_conv1')
        first = tl.batch_normalization(net,trainable=is_train,name='one_feature')
        first = tf.nn.relu(first)
        net = tl.max_pooling2d(net,(2,2),(2,2),name='transition2_pool1')

        ##DenseBlock 3   8x
        net = dense_block(net,iter=8,two_conv=48,one_conv=192,name='denseblock3')

        #transition w/o pooling layer 1
        net = tl.batch_normalization(net,trainable=is_train,name='transition_w_o1_bn')
        net = tf.nn.relu(net)
        net = tl.conv2d(net,1184,(1,1),padding='same',name='transition_pooling1')

        #Dense block 4  8x
        net = dense_block(net,iter=8,two_conv=48,one_conv=192,is_train = is_train,name='denseblock4')

        ##transition w/o pooling layer 2
        net = tl.batch_normalization(net,trainable=is_train,name='transition_w_o2_bn')
        net = tf.nn.relu(net)
        net = tl.conv2d(net,256,(1,1),padding='same',name='transition_pooling2')
        net2 = tl.max_pooling2d(first,(2,2),(2,2))
        net2 = tl.batch_normalization(net2,trainable=is_train,name='bn_3')
        net2 = tf.nn.relu(net2)
        net2 = tl.conv2d(net2,256,(1,1),padding='same',name='d_cnnb')
        net = tf.concat([net,net2],axis=-1)
        second = tl.batch_normalization(net,trainable=is_train,name='bn_4')
        second = tf.nn.relu(second)

        net = denseblock_pl(net,step=256,channel=256,is_train = is_train,name='densepl_1')
        third = tl.batch_normalization(net,trainable=is_train,name='bn_5')
        third = tf.nn.relu(third)

        net = denseblock_pl(net, step=128, channel=128, is_train=is_train, name='densepl_2')
        fourth = tl.batch_normalization(net, trainable=is_train, name='bn_6')
        fourth = tf.nn.relu(fourth)

        net = denseblock_pl(net, step=128, channel=128, is_train=is_train, name='densepl_3')
        fifth = tl.batch_normalization(net, trainable=is_train, name='bn_7')
        fifth = tf.nn.relu(fifth)

        net = denseblock_pl(net, step=128, channel=128, is_train=is_train, name='densepl_4',padding='valid')
        sixth = tl.batch_normalization(net, trainable=is_train, name='bn_8')
        sixth = tf.nn.relu(sixth)



        out1 = tl.conv2d(first,6 * 25,(3,3),padding='same',name='first')
        out2 = tl.conv2d(second,6 * 25,(3,3),padding='same',name='second')
        out3 = tl.conv2d(third,6 * 25,(3,3),padding='same',name='third')
        out4 = tl.conv2d(fourth,6 * 25,(3,3),padding='same',name='fourth')
        out5 = tl.conv2d(fifth,6 * 25,(3,3),padding='same',name='fifth')
        out6 = tl.conv2d(sixth,6 * 25,(3,3),padding='same',name='sixth')

        feats = [out1,out2,out3,out4,out5,out6]
        feature_map = []
        for i,feat in enumerate(feats):
            s = feat.get_shape()[1]
            change = tf.reshape(feat,[-1,s * s * 6,25])
            feature_map.append(change)
        feature_map = tf.concat(feature_map,axis=1)
        # print(feature_map)
        cls = feature_map[:,:,:21]
        loc = feature_map[:,:,21:]

        return cls,loc






if __name__ == '__main__':
    x = tf.placeholder(dtype=tf.float32,shape = [None,300,300,3])
    main_net(x,False)






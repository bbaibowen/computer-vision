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
        net = tl.conv2d(net,416,(1,1),(1,1),padding='same',name='transition1_conv1')
        net = tl.max_pooling2d(net,(2,2),(2,2),padding='same',name='transition1_pool1')

        #DenseBlock 2   8x
        net = dense_block(net,8,48,192,is_train = False,name='denseblock2')

        #Transition Layer 2
        net = tl.batch_normalization(net,trainable=is_train,name='transition2_bn')
        net = tl.conv2d(net,800,(1,1),(1,1),padding='same',name='transition2_conv1')
        net = tl.max_pooling2d(net,(2,2),(2,2),name='transition2_pool1')

        ##DenseBlock 3   8x
        net = dense_block(net,iter=8,two_conv=48,one_conv=192,name='denseblock3')

        #transition w/o pooling layer 1
        net = tl.conv2d(net,1184,(1,1),padding='same',name='transition_pooling1')

        #Dense block 4  8x
        net = dense_block(net,iter=8,two_conv=48,one_conv=192,is_train = is_train,name='denseblock4')

        ##transition w/o pooling layer 2
        net = tl.conv2d(net,1568,(1,1),padding='same',name='transition_pooling2')
        print(net)







if __name__ == '__main__':
    x = tf.placeholder(dtype=tf.float32,shape = [None,300,300,3])
    main_net(x,False)






import tensorflow as tf
slim = tf.contrib.slim


def Vgg():

    input = tf.placeholder(dtype = tf.float32,shape=[None,600,800,3])

    net = slim.conv2d(input, 64, [3, 3], trainable=False, scope='conv1_1')
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


    return net,input
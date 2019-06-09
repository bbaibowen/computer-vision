import tensorflow as tf
import numpy as np
slim = tf.contrib.slim


def PRelu(x):

    alpha = tf.get_variable('alphas',shape = x.get_shape()[-1],dtype = tf.float32)
    out = tf.nn.relu(x) + tf.multiply(alpha,-tf.nn.relu(-x))

    return out

def PNet(x):
    with tf.variable_scope('PNet'):
        with slim.arg_scope([slim.conv2d], activation_fn=PRelu,
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005), padding='VALID'):

            net = slim.conv2d(x,10,3,scope = 'conv1')
            net = slim.max_pool2d(net,[3,3],2,padding = 'SAME',scope='pool1')
            net = slim.conv2d(net,16,3,scope = 'conv2')
            net = slim.conv2d(net,32,3,scope = 'conv3')

            cls = slim.conv2d(net,2,1,activation_fn = tf.nn.softmax,scope = 'conv4_1')
            box = slim.conv2d(net,4,1,activation_fn = None,scope = 'conv4_2')
            landmark = slim.conv2d(net,10,1,activation_fn = None,scope = 'conv4_3')

            cls = tf.squeeze(cls,axis = 0)
            box = tf.squeeze(box,axis = 0)
            landmark = tf.squeeze(landmark,axis=0)

            return cls,box,landmark


def RNet(x):
    with tf.variable_scope('RNet'):
        with slim.arg_scope([slim.conv2d], activation_fn=PRelu,
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005), padding='VALID'):
            net = slim.conv2d(x, 28, 3, scope='conv1')
            net = slim.max_pool2d(net, [3, 3], 2, padding='SAME', scope='pool1')
            net = slim.conv2d(net, 48, 3, scope='conv2')
            net = slim.max_pool2d(net, [3, 3], 2, scope='pool2')
            net = slim.conv2d(net, 64, 2, scope='conv3')
            net = slim.flatten(net)
            net = slim.fully_connected(net, num_outputs=128, scope='fc1')

            cls = slim.fully_connected(net, num_outputs=2, activation_fn=tf.nn.softmax, scope='cls_fc')
            box = slim.fully_connected(net, num_outputs=4, activation_fn=None, scope='bbox_fc')
            landmark = slim.fully_connected(net, num_outputs=10, activation_fn=None, scope='landmark_fc')


            return cls,box,landmark


def ONet(x):
    with tf.variable_scope('ONet'):
        with slim.arg_scope([slim.conv2d], activation_fn=PRelu,
                            weights_initializer=slim.xavier_initializer(),
                            weights_regularizer=slim.l2_regularizer(0.0005), padding='VALID'):
            net = slim.conv2d(x, 32, 3, scope='conv1')
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, padding='SAME', scope='pool1')
            net = slim.conv2d(net, 64, 3, scope='conv2')
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool2')
            net = slim.conv2d(net, 64, 3, scope='conv3')
            net = slim.max_pool2d(net, kernel_size=[3, 3], stride=2, scope='pool3', padding='SAME')
            net = slim.conv2d(net, 128, 2, scope='conv4')
            net = slim.flatten(net)
            net = slim.fully_connected(net, num_outputs=256, scope='fc1')

            cls = slim.fully_connected(net, num_outputs=2, activation_fn=tf.nn.softmax, scope='cls_fc')
            box = slim.fully_connected(net, num_outputs=4, activation_fn=None, scope='bbox_fc')
            landmark = slim.fully_connected(net, num_outputs=10, activation_fn=None, scope='landmark_fc')

            return cls, box, landmark


if __name__ == '__main__':
    x = tf.placeholder(dtype=tf.float32,shape = [None,12,12,3])
    cls, box, landmark = PNet(x)
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess,'PNet/PNet-30')
    print('done')
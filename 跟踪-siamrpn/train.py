from load_data import Reader
from network import Siam_RPN
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

num_data = 21455
MAX_STEP = num_data * 30
decay_step = int(num_data / 4)
EPOCH = 0
DATA_PATH = 'vot2016'


data_reader = Reader(DATA_PATH)
# anchors = gen_anchors()
target,_,search,gt_box,_,_ = data_reader.get_batch()
# print(target,search,gt_box)

# print(gt_box.shape,gt_box[0].shape)
network = Siam_RPN()
cls,reg = network.build_train(target,search)
loss_total,ts = network.loss(gt_box[0],cls,reg)


saver = tf.train.Saver()
# with tf.name_scope('op'):
#     global_step = tf.Variable(0, trainable=False)
#     rate = tf.train.exponential_decay(0.001, global_step, decay_step, 0.95, staircase=True, name='lr')
#     op = tf.train.GradientDescentOptimizer(rate).minimize(loss_total,global_step=global_step)



op = tf.train.AdamOptimizer().minimize(loss_total)

coord=tf.train.Coordinator()
sess=tf.Session()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)
sess.run(tf.global_variables_initializer())

# _index_ = []
_loss_ = []
for i in range(MAX_STEP):
    # _index_.append(i)
    _,loss_value = sess.run([op,loss_total])
    print(sess.run([cls,reg,search]))

    print('step={},loss={}'.format(i, loss_value))

    # if i % 100 == 0 and i != 0:
    #     plt.plot(_loss_)
    #     plt.savefig('./ckpt/img/{}.jpg'.format(i))
    #     plt.show()


    if i == 200:
        EPOCH += 1
        saver.save(sess,'./ckpt/model',EPOCH)
        print('EPCHO = {} \n save it Successful!'.format(EPOCH))





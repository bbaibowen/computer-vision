import tensorflow as tf
from data_load import get_voc,anchor_target_layers
from dsod_net import main_net
from gen_anchors import ANCHORS
from losses import loss

epoch = 0
MAX_STEP = 16000 * 30
imageinput = tf.placeholder(tf.float32, [None, 300, 300, 3])

gt_cls = tf.placeholder(shape=[None, 11640], dtype=tf.int32,
                                       name='groundtruth_class')
gt_loc = tf.placeholder(shape=[None, 11640, 4], dtype=tf.float32,
                                      name='groundtruth_location')
pos = tf.placeholder(shape=[None, 11640], dtype=tf.float32,
                                       name='groundtruth_positives')
neg = tf.placeholder(shape=[None, 11640], dtype=tf.float32,
                                       name='groundtruth_negatives')

groundtruth_count = tf.add(pos, neg)

cls,loc = main_net(imageinput,True)
_loss = loss(cls,loc,gt_cls,gt_loc,pos,groundtruth_count)

with tf.name_scope('train_op'):
    global_step = tf.Variable(0, trainable=False)
    lr = tf.train.exponential_decay(0.001, global_step, int(16000/4), 0.95, staircase=True, name='lr')
    op = tf.train.GradientDescentOptimizer(lr).minimize(_loss,global_step=global_step)

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(MAX_STEP):
    train_data, actual_data = get_voc(5)
    gt_class, gt_location, gt_positives, gt_negatives = anchor_target_layers(actual_data,ANCHORS)
    feed_dict = {
        imageinput:train_data,
        gt_cls:gt_class,
        gt_loc:gt_location,
        pos:gt_positives,
        neg:gt_negatives
    }
    _,loss_total,_ = sess.run([op,_loss,lr],feed_dict=feed_dict)
    print(loss_total)

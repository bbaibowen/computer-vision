import os
import random

import tensorflow as tf

import vggnet
from configs import *
from datas import get_data
from network import Network

DATA_PATH = 'C:\\Users\\ZD\\Desktop\\FasterRCNN_TF-master\\FasterRCNN_TF-master\\experiments\\experiment1\\data'
XML_PATH = DATA_PATH + '\\xmls\\'
IMAGE_PATH = DATA_PATH + '\\images\\'

def train():

    fasterRCNN = Network()
    fasterRCNN.build(is_training=True)
    train_op = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9).minimize(fasterRCNN._losses['total_loss'])
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init_op)

        train_img_names = os.listdir(IMAGE_PATH)
        trained_times = 0

        for epoch in range(1, MAX_EPOCH + 1):
            random.shuffle(train_img_names)
            for train_img_name in train_img_names:
                boxes, img = get_data(train_img_name)

                sess.run(train_op, feed_dict={fasterRCNN.input: img, fasterRCNN.gt_boxes: boxes})

                trained_times += 1

                total_loss = sess.run(fasterRCNN._losses['total_loss'],
                                      feed_dict={fasterRCNN.input: img, fasterRCNN.gt_boxes: boxes})
                print(total_loss)

                if epoch % 10 == 0:
                    save_path = saver.save(sess, os.path.join(CHECKPOINTS_PATH, "model_" + str(epoch) + ".ckpt"))

            save_path = saver.save(sess, os.path.join(CHECKPOINTS_PATH, "model_final.ckpt"))


if __name__ == '__main__':
    train()
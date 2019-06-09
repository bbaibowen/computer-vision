import tensorflow as tf
import numpy as np
import cv2
import net
from load_data import Generator






IMG_SHAPE = 227
MAX_STEP = 1000
BATCH = 10
DATA_PATH = 'vot2016'
TARGET_PATH = 'list.txt'


data_manager = Generator(DATA_PATH,target_list=TARGET_PATH, batch_size=BATCH)

# img_tensor = tf.placeholder(dtype=tf.float32,shape=[BATCH,IMG_SHAPE,IMG_SHAPE,3])
# target_tensor = tf.placeholder(dtype=tf.float32,shape=[BATCH,IMG_SHAPE,IMG_SHAPE,3])
gt_box = tf.placeholder(tf.float32,shape=[BATCH,4])

test = net.goturn(True)
fc_four, x, target = test.main_net()
loss = test.loss(fc_four,gt_box)

# global_step = tf.Variable(0,False,name='global_step')
op = tf.train.AdamOptimizer(1e-3).minimize(loss)
init = tf.global_variables_initializer()
# init_local = tf.local_variables_initializer()

sess = tf.Session()
sess.run(init)
# sess.run(init_local)
saver = tf.train.Saver()
# saver.restore(sess,'C:\\Users\\ZD\\Desktop\\Object-Tracking-GOTURN-develop\\ck\\checkpoint.ckpt-1')

for i in range(MAX_STEP):
    img, y = data_manager.get_data(i)
    target_img,search_img = img

    _,loss_total = sess.run([op,loss],feed_dict={
                                  x:search_img,
                                  target:target_img,
                                  gt_box:y
                                  })

    print(loss_total)
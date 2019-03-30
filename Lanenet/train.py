import tensorflow as tf
import numpy as np
from lanenet import Lanenet


'''
这里我需要提前说一下，因为我的电脑内存不够了，所以训练的数据我就没下，你们可以到官网去下数据集，一共10G，
数据集是tusimple，

加载数据的代码直接用源码就可以了，我一起放在文件中

该模型训练需要三类图片分别是原始图片、二值化分割图像（255 表示车道区域，0 表示其他的）和实例图像（标签图像）。
原始图像放在 data 目录下的 images 目录下
二值化分割图像放在 data 目录下的 binary 目录下
实例图像（标签图像）放在 data 目录下的 labels 目录下


然后需要在 data 目录下放 train.txt 文件夹，文件中每一行存放的是原始图像存放的路径（包含图像名称）、二值化图像存放的路径（包含图像名称）
和实例图像（标签图像）存放的路径（包含图像名称）。这里顺序要与教程中的
一致。
举 例 ： /XXX/lanenet/data/training/images/170927_063811892_Camera_5.jpg
/XXX/lanenet/data/training/binary/170927_063811892_Camera_5_bin.png
/XXX/lanenet/data/training/labels/170927_063811892_Camera_5_bin.png

lanenet_data_processor.py ：为训练模型提供输入图片数据
'''
MAX_STEP = 100000
BATCH_SIZE = 50
TRAIN_DATA_PATH = ''


img_src = tf.placeholder(tf.float32,[BATCH_SIZE,256,512,3])
binary_label_tensor = tf.placeholder(tf.int64,[BATCH_SIZE,256,512,1])
instance_label_tensor = tf.placeholder(tf.float32,[BATCH_SIZE,256,512])

network = Lanenet('train')

deconv_final, logits = network.build_model(img_src)

loss = network.loss(binary_label_tensor,instance_label_tensor,logits,deconv_final,'lanenet_model')

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(0.1,global_step,100000,0.1,True)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=0.9).minimize(loss=loss,var_list=tf.trainable_variables(),global_step=global_step)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    #后续就比较简单了，加载数据，做预处理，收集参数等等
    print(loss)






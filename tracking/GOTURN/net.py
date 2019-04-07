import tensorflow as tf
import numpy as np
import cv2



# from handle import Bbox,Frame

class goturn(object):

    def __init__(self,is_train):
        self.img_size = (227,227)
        self.mean = [104, 117, 123]
        self.is_train = is_train


    def conv2d(self,x,weights,strides,pad = 0,trainable = False, name = None,iter = 1,bias_init = 0.0):

        with tf.variable_scope(name):

            if pad > 0:
                x = tf.pad(x,[[0,0],[pad,pad],[pad,pad],[0,0]])

            weight = tf.Variable(tf.truncated_normal(weights,dtype=tf.float32,stddev=1e-2),trainable = trainable,name='weights')
            biases = tf.Variable(tf.constant(0.1,shape=[weights[3]],dtype=tf.float32),trainable=trainable,name='biases')

            if iter == 1:
                net = tf.nn.bias_add(tf.nn.conv2d(x,weight,strides,padding='VALID'),biases)
            elif iter == 2:
                w1,w2 = tf.split(weight,2,axis=3)
                x1,x2 = tf.split(x,2,axis=3)
                conv_one = tf.nn.conv2d(x1,w1,strides,padding='VALID')
                conv_two = tf.nn.conv2d(x2,w2,strides,padding='VALID')
                net = tf.nn.bias_add(tf.concat([conv_one,conv_two],axis=3),biases)

            net = tf.nn.relu(net)

            return net


    def caffenet(self,x,name):

        net = self.conv2d(x, [11, 11, 3, 96], [1, 4, 4, 1], name=name + '_conv_1')

        net = tf.nn.max_pool(net, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name=name + '_pool1')

        net = tf.nn.local_response_normalization(net, depth_radius=2, alpha=1e-4, beta=0.75, name=name +'_lrn1')

        net = self.conv2d(net, [5, 5, 48, 256], [1, 1, 1, 1], pad=2, iter=2, bias_init=1.0, name=name + '_conv_2')

        net = tf.nn.max_pool(net, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name=name + '_pool2')

        net = tf.nn.local_response_normalization(net, depth_radius=2, alpha=1e-4, beta=0.75, name=name + '_lrn2')

        net = self.conv2d(net, [3, 3, 256, 384], [1, 1, 1, 1], pad=1, name=name + '_conv_3')

        net = self.conv2d(net, [3, 3, 192, 384], [1, 1, 1, 1], pad=1, iter=2, name=name + '_conv_4')

        net = self.conv2d(net, [3, 3, 192, 256], [1, 1, 1, 1], pad=1, iter=2, bias_init=1.0, name=name + '_conv_5')

        net = tf.nn.max_pool(net, [1, 3, 3, 1], [1, 2, 2, 1], padding='VALID', name=name + '_pool5')

        return net




    def fc_layer(self,x,c,name,af = True):
        with tf.variable_scope(name):
            shape = int(np.prod(x.get_shape()[1:]))
            w = tf.Variable(tf.truncated_normal([shape,c],dtype=tf.float32,stddev=0.001),name = 'weights')
            b = tf.Variable(tf.constant(0.1,shape=[c],dtype=tf.float32),name = 'biases')
            x = tf.reshape(x,[-1,shape])
            out = tf.nn.bias_add(
                tf.matmul(x,w),b
            )
            if af:
                return tf.nn.relu(out)
            else:
                return out

    def main_net(self):

        x = tf.placeholder(tf.float32, [None, 227, 227, 3])
        target = tf.placeholder(tf.float32, [None, 227, 227, 3])


        target_net = self.caffenet(target,'target')
        img_net = self.caffenet(x,'image')
        net = tf.concat([target_net,img_net],axis=3)
        net = tf.transpose(net,(0,3,1,2))
        fc_one = self.fc_layer(net,4096,'fc1')
        if self.is_train:
            fc_one = tf.nn.dropout(fc_one,.5)
        fc_two = self.fc_layer(fc_one,4096,'fc2')
        if self.is_train:
            fc_two = tf.nn.dropout(fc_two,.5)
        fc_three = self.fc_layer(fc_two, 4096, 'fc3')
        if self.is_train:
            fc_three = tf.nn.dropout(fc_three, .5)
        fc_four = self.fc_layer(fc_three,4,'fc4',False)

        return fc_four,x,target


    def loss(self,fc_four,box_gt):
        a = tf.subtract(fc_four,box_gt)
        al_abs = tf.abs(tf.reshape(a,[-1]))
        loss = tf.reduce_sum(al_abs,name='loss')

        return loss




class Box:
    def __init__(self,box):
        self.x1, self.y1, self.x2, self.y2 = box
        self.scaling = 2
        self.x = (self.x1 + self.x2) / 2
        self.y = (self.y1 + self.y2) / 2
        self.w = max(1.,self.scaling * (self.x2 - self.x1))
        self.h = max(1.,self.scaling * (self.y2 - self.y1))

    def update_xywh(self):
        self.x = (self.x1 + self.x2) / 2
        self.y = (self.y1 + self.y2) / 2
        self.w = max(1., self.scaling * (self.x2 - self.x1))
        self.h = max(1., self.scaling * (self.y2 - self.y1))



    def get_img(self,frame):
        self.scaling_x1 = max(1, int(self.x - (self.w / 2)))
        self.scaling_x2 = int(self.x + (self.w / 2))
        self.scaling_y1 = max(1, int(self.y - (self.h / 2)))
        self.scaling_y2 = int(self.y + (self.h / 2))
        if self.scaling_x2 > frame.shape[1]:
            self.scaling_x2 = frame.shape[1]
        if self.scaling_y2 > frame.shape[0]:
            self.scaling_y2 = frame.shape[0]
        self.img = frame[self.scaling_y1:self.scaling_y2,self.scaling_x1:self.scaling_x2]

        return cv2.resize(self.img,(227,227))

    def update(self, pred_box):

        offset = [(pred_box[0] - .25) * self.img.shape[1],
                  (pred_box[2] - .75) * self.img.shape[1],
                  (pred_box[1] - .25) * self.img.shape[0],
                  (pred_box[3] - .75) * self.img.shape[0]]  # x1,x2,y1,y2

        self.x1 += offset[0]
        self.x2 += offset[1]
        self.y1 += offset[2]
        self.y2 += offset[3]
        self.update_xywh()

    def draw(self,frame):
        p1 = (int(self.x1), int(self.y1))
        p2 = (int(self.x2), int(self.y2))
        cv2.rectangle(frame, p1, p2, (255, 255, 255), 3, 2)


def test_ma():
    video_path = 'C:\\Users\\ZD\\Desktop\\Object-Tracking-GOTURN-develop\\video_test2.mp4'

    test = goturn(False)
    ckpt_path = 'C:\\Users\\ZD\\Desktop\\Object-Tracking-GOTURN-develop\\ck\\checkpoint.ckpt-1'

    video = cv2.VideoCapture(video_path)

    # 这里我们默认取第一帧

    _, frame = video.read()
    bbox = cv2.selectROI(frame, False)
    bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

    bbox = Box(bbox)
    target_img = bbox.get_img(frame)

    fc_four, x, target = test.main_net()
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, ckpt_path)

    while True:
        _, frame = video.read()
        if not _:
            break
        test_img = bbox.get_img(frame)
        prediction = sess.run(fc_four, feed_dict={x: np.expand_dims(test_img, 0),
                                                  target: np.expand_dims(target_img, 0)})
        bbox.update(prediction[0] / 10)

        bbox.draw(frame)
        target_img = bbox.get_img(frame)

        cv2.imshow('GOTURN', frame)
        if cv2.waitKey(1) & 0xff == 27:
            break

if __name__ == '__main__':

    test_ma()
import tensorflow as tf
import cv2
from net import PNet,RNet,ONet
import numpy as np
from utils import process_img,generate_box,NMS,change_box,pad,calibrate_box


class Pred_PNet:

    def __init__(self):
        graph = tf.Graph()
        with graph.as_default():
            self.factor = 0.7   #缩放系数的增长比例
            self.min_face = 24
            self.pnet_size = 12
            #默认每次只预测一张
            self.P_input = tf.placeholder(dtype = tf.float32,shape = [1,None,None,3])
            self.score,self.box,_ = PNet(self.P_input)
            self.sess = tf.Session()
            self.saver = tf.train.Saver()


    def load_pnet(self,input,model_path = 'PNet/PNet-30'):

        self.saver.restore(self.sess,model_path)
        pred_scores,pred_box = self.sess.run([self.score,self.box],feed_dict={self.P_input:input})

        return pred_scores,pred_box

    def p_pent(self,img):

        scale = float(self.pnet_size / self.min_face)
        _img = process_img(img,scale)
        h,w,_ = _img.shape
        all_boxes = []

        while min(h,w) > self.pnet_size:

            # print(_img.shape)
            p_cls,p_box = self.load_pnet(np.expand_dims(_img,axis=0))

            boxes = generate_box(p_cls[:,:,1],p_box,scale,0.6)
            scale *= self.factor
            _img = process_img(img,scale)
            h,w,_ = _img.shape

            nms = NMS(boxes[:,:5],0.5)
            boxes = boxes[nms]

            all_boxes.append(boxes)

        all_boxes = np.vstack(all_boxes)
        # box = all_boxes[:,:5]
        box_w = all_boxes[:,2] - all_boxes[:,0]
        box_h = all_boxes[:,3] - all_boxes[:,1]

        res_boxes = np.vstack([
            all_boxes[:,0] + all_boxes[:,5] * box_w,
            all_boxes[:,1] + all_boxes[:,6] * box_h,
            all_boxes[:,2] + all_boxes[:,7] * box_w,
            all_boxes[:,3] + all_boxes[:,8] * box_h,
            all_boxes[:,4]
        ])#[5,NUM]  --->  [NUM,5]

        print(res_boxes.shape)
        res_boxes = res_boxes.T
        print(res_boxes.shape)

        return res_boxes

class Pred_RNet:

    def __init__(self):
        graph = tf.Graph()
        with graph.as_default():
            self.R_input = tf.placeholder(dtype=tf.float32,shape=[None,24,24,3])
            self.r_score,self.r_box,_ = RNet(self.R_input)
            self.sess = tf.Session()
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess,'RNet/RNet-22')




    def load_rnet(self,img):


        score,box = self.sess.run([self.r_score,self.r_box],feed_dict={self.R_input:img})
        return score,box

    def p_rent(self,img,bboxes):

        h,w,_ = img.shape
        bboxes = change_box(bboxes)

        bboxes[:,0:4] = np.round(bboxes[:,0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(bboxes,w,h)
        # print(edy - dy == ey - y)

        match_size = np.ones_like(tmpw) * 24
        zeros = np.zeros_like(tmpw)
        ones = np.ones_like(tmpw)
        num = np.sum(
            np.where((np.minimum(tmpw,tmph) >= match_size),ones,zeros)
        )
        _img = np.zeros((num,24,24,3),dtype=np.float32)


        for i in range(num):

            tmp = np.zeros((tmph[i],tmph[i],3),dtype=np.float32)
            tmp[dy[i]:edy[i],dx[i]:edx[i],:] = img[y[i]:ey[i],x[i]:ex[i],:]
            _img[i,:,:,:] = (cv2.resize(tmp,(24,24)) - 127.5) / 127.5

        cls_score,box = self.load_rnet(_img)
        cls_score = cls_score[:,1]
        keep_index = np.where(cls_score > 0.7)[0]
        boxes = bboxes[keep_index]
        boxes[:,4] = cls_score[keep_index]
        box = box[keep_index]


        nms = NMS(boxes,0.7)
        boxes = boxes[nms]
        box = box[nms]

        result = calibrate_box(boxes,box)

        return result

class Pred_Onet:

    def __init__(self):
        graph = tf.Graph()
        with graph.as_default():
            self.O_input = tf.placeholder(dtype=tf.float32, shape=[None, 48, 48, 3])
            self.o_cls_prob, self.o_bbox_pred, self.o_landmark_pred = ONet(self.O_input)
            self.sess = tf.Session()
            self.saver = tf.train.Saver()
            self.saver.restore(self.sess, 'ONet/ONet-14')

    def load_onet(self,img):
        cls_prob, box, landmark = self.sess.run([self.o_cls_prob, self.o_bbox_pred, self.o_landmark_pred],
                                                feed_dict={self.O_input: img})

        return cls_prob, box, landmark

    def o_onet(self,img,bboxes):

        h, w, c = img.shape
        bboxes = change_box(bboxes)
        bboxes[:, 0:4] = np.round(bboxes[:, 0:4])


        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(bboxes, w, h)
        crop_img = np.zeros((bboxes.shape[0], 48, 48, 3), dtype=np.float32)
        for i in range(bboxes.shape[0]):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i], dx[i]:edx[i], :] = img[y[i]:ey[i], x[i]:ex[i], :]
            crop_img[i, :, :, :] = (cv2.resize(tmp, (48, 48)) - 127.5) / 128
        scores, box, landmark = self.load_onet(crop_img)
        scores = scores[:, 1]
        keep_index = np.where(scores > 0.7)[0]
        bboxes = bboxes[keep_index]
        bboxes[:, 4] = scores[keep_index]
        box = box[keep_index]
        landmark = landmark[keep_index]

        w = bboxes[:, 2] - bboxes[:, 0]
        h = bboxes[:, 3] - bboxes[:, 1]
        landmark[:, 0::2] = (np.tile(w, (5, 1)) * landmark[:, 0::2].T + np.tile(bboxes[:, 0], (5, 1))).T
        landmark[:, 1::2] = (np.tile(h, (5, 1)) * landmark[:, 1::2].T + np.tile(bboxes[:, 1], (5, 1))).T

        res_boxes = calibrate_box(bboxes, box)

        nms = NMS(res_boxes, 0.6)
        res_boxes = res_boxes[nms]
        landmark = landmark[nms]

        return res_boxes, landmark

def detect_face(img):
    # img = cv2.imread(img_path)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    test1 = Pred_PNet()
    res_box = test1.p_pent(img)
    # print(res_box)
    test2 = Pred_RNet()
    R_box = test2.p_rent(img, res_box)
    print(R_box.shape)

    test3 = Pred_Onet()
    res_b, landmark = test3.o_onet(img, R_box)

    return res_b



if __name__ == '__main__':

    img = cv2.imread('1.png')
    test1 = Pred_PNet()
    res_box = test1.p_pent(img)
    # print(res_box)
    test2 = Pred_RNet()
    R_box = test2.p_rent(img,res_box)
    print(R_box.shape)

    test3 = Pred_Onet()
    res_b,landmark = test3.o_onet(img,R_box)
    print(res_b.shape)
    # res_b, landmark = detect_face('2.png')

    for i in range(res_b.shape[0]):

        score = res_b[i,4]
        box = res_b[i,:4]
        p1 = (int(box[0]),int(box[1]))
        p2 = (int(box[2]),int(box[3]))
        cv2.rectangle(img,p1,p2,(255,255,255),2)
        cv2.putText(img,'{:.2f}'.format(score),(int(box[0]),int(box[1]) - 3),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,9),2)

    for i in range(landmark.shape[0]):
        for j in range(len(landmark[i]) // 2):
            cv2.circle(img, (int(landmark[i][2 * j]), int(int(landmark[i][2 * j + 1]))), 2, (0, 255, 0))

    cv2.imshow('test',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
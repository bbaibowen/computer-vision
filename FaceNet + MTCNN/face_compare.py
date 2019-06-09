import tensorflow as tf
import numpy as np
from mtnet import detect_face
from facenet_net import inference
import cv2

FACENET_MODEL_PATH = 'facenet_model/model-20170512-110547.ckpt-250000'

def find_face_for_pictures(img):

    boxes = detect_face(img)
    num_box = boxes.shape[0]
    bboxes = []
    crop_imgs = []
    if num_box > 0:
        det = boxes[:,:4]
        det_arr = []
        img_size = np.asarray(img.shape)[:2]
        for i in range(num_box):
            det_arr.append(np.squeeze(det[i]))

        for i,det in enumerate(det_arr):
            det = np.squeeze(det)
            bb=[int(max(det[0],0)), int(max(det[1],0)), int(min(det[2],img_size[1])), int(min(det[3],img_size[0]))]
            cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),(255,255,255),2)
            bboxes.append((bb[0],bb[1]))
            crop_img = img[bb[1]:bb[3],bb[0]:bb[2],:]
            crop_img = cv2.resize(crop_img,(160,160))
            crop_img = cv2.cvtColor(crop_img,cv2.COLOR_BGR2RGB)
            crop_imgs.append(crop_img)
        return img,crop_imgs,bboxes




def face_Net(crop_imgs1,crop_imgs2):
    graph = tf.Graph()
    with graph.as_default():
        img_holder = tf.placeholder(dtype=tf.float32,shape=[None,160,160,3])
        embeddings,end_ponits = inference(img_holder,0.5,False)
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess,FACENET_MODEL_PATH)
        embeddings1 = sess.run(embeddings,feed_dict={img_holder:np.expand_dims(crop_imgs1[0],axis=0)})
        embeddings2 = sess.run(embeddings,feed_dict={img_holder:np.expand_dims(crop_imgs2[0],axis=0)})

        return embeddings1,embeddings2







if __name__ == '__main__':

    img2 = cv2.imread('1.png')
    im = cv2.imread('1.jpg')
    img1, crop_imgs1, bboxes1 = find_face_for_pictures(im)
    img2,crop_imgs2,bboxes2 = find_face_for_pictures(img2)
    e1,e2 = face_Net(crop_imgs1,crop_imgs2)
    distance = np.mean(np.square(e1 - e2),axis=1)
    print(distance)
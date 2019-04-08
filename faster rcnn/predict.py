import os
import cv2
from configs import *
import numpy as np
import tensorflow as tf
import vggnet
from network import Network

# from data_loaders.data import get_predict_data


def bbox_transform_inv(boxes, deltas):
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    ## 0::4表示先取第一个元素，以后每4个取一个
    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]

    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]
    ## 预测后的（x1,y1,x2,y2）存入 pred_boxes
    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def clip_boxes(boxes, im_shape):
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def visualization(result_list, path=IMAGE_RESULT_PATH):
    file_name = result_list[0][0]
    im = cv2.imread(os.path.join(PREDICT_IMG_DATA_PATH, file_name))

    for result in result_list:
        class_name = result[1]
        score = result[2]
        x1 = result[3]
        y1 = result[4]
        x2 = result[5]
        y2 = result[6]
        cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        # 各参数依次是：图片，添加的文字，左上角坐标，字体，字体大小，颜色，字体粗细
        cv2.putText(im, str(class_name) + ': ' + str(score), (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8,
                    (0, 0, 255), 2)
    cv2.imwrite(os.path.join(path, file_name), im)


def write_txt_result(result_list, path=TXT_RESULT_PATH):
    file_name = result_list[0][0].split('.')[0]
    txt_file = open(os.path.join(path, file_name + '.txt'), 'w')
    for result in result_list:
        for element in result:
            txt_file.write(str(element) + ' ')
        txt_file.write('\n')
    txt_file.close()


def nms(detections, thresh):
    """Pure Python NMS baseline."""
    x1 = detections[:, 0]
    y1 = detections[:, 1]
    x2 = detections[:, 2]
    y2 = detections[:, 3]
    scores = detections[:, 4]
    ## 单个框面积大小
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    ## 按照得分值从大到小将序号排列
    order = scores.argsort()[::-1]  ## [::-1]倒序

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)  ## 得分最大的保留，保留值为序号

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def predict():
    fasterRCNN = Network()
    fasterRCNN.build(is_training=False)
    features, img_holder = vggnet.Vgg()
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    # init_loc = tf.local_variables_initializer()
    with tf.Session() as sess:

        saver.restore(sess, os.path.join(CHECKPOINTS_PATH, "VGGnet_fast_rcnn_iter_70000.ckpt"))
        print("Model restored.")
        # base_extractor = VGG16(include_top=False)
        # extractor = Model(inputs=base_extractor.input, outputs=base_extractor.get_layer('block5_conv3').output)
        predict_img_names = os.listdir(PREDICT_IMG_DATA_PATH)

        for predict_img_name in predict_img_names:
            img_data, img_info = get_predict_data(predict_img_name)
            print(img_data.shape,img_info)
            # features = extractor.predict(img_data, steps=1)
            # features,img_holder = vggnet.Vgg()
            # sess.run(init)
            features = sess.run(features,feed_dict={img_holder:img_data})
            rois, scores, regression_parameter = sess.run(
                [fasterRCNN._predictions["rois"], fasterRCNN._predictions["cls_prob"],
                 fasterRCNN._predictions["bbox_pred"]],
                feed_dict={fasterRCNN.feature_map: features,
                           fasterRCNN.image_info: img_info})

            boxes = rois[:, 1:5] / img_info[2]
            scores = np.reshape(scores, [scores.shape[0], -1])
            regression_parameter = np.reshape(regression_parameter, [regression_parameter.shape[0], -1])
            pred_boxes = bbox_transform_inv(boxes, regression_parameter)
            pred_boxes = clip_boxes(pred_boxes, [img_info[0] / img_info[2], img_info[1] / img_info[2]])

            result_list = []
            for class_index, class_name in enumerate(CLASSES[1:]):
                class_index += 1  # 因为跳过了背景类别
                cls_boxes = pred_boxes[:, 4 * class_index:4 * (class_index + 1)]  # TODO:
                cls_scores = scores[:, class_index]
                detections = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
                keep = nms(detections, NMS_THRESH)
                detections = detections[keep, :]

                inds = np.where(detections[:, -1] >= CONF_THRESH)[0]  # 筛选结果
                for i in inds:
                    result_for_a_class = []
                    bbox = detections[i, :4]
                    score = detections[i, -1]
                    result_for_a_class.append(predict_img_name)
                    result_for_a_class.append(class_name)
                    result_for_a_class.append(score)
                    for coordinate in bbox:
                        result_for_a_class.append(coordinate)
                    result_list.append(result_for_a_class)
                    # result_for_a_class = [fileName,class_name,score,x1,y1,x2,y2]
            # if len(result_list) == 0:
            #     continue

            if TXT_RESULT_WANTED:
                write_txt_result(result_list)

            if IS_VISIBLE:
                visualization(result_list)


if __name__ == '__main__':
    predict()

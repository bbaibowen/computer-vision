import xml.etree.ElementTree as etxml
import numpy as np
import random
import skimage.io
import skimage.transform
import tensorlayer as tl
import os
import math



lable_arr = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
file_name_list = os.listdir('./VOC2007/JPEGImages/')
threshold = 0.6
num_anchors = 11640

def get_voc(batch_size):
    def get_actual_data_from_xml(xml_path):
        actual_item = []
        try:
            annotation_node = etxml.parse(xml_path).getroot()
            img_width = float(annotation_node.find('size').find('width').text.strip())
            img_height = float(annotation_node.find('size').find('height').text.strip())
            object_node_list = annotation_node.findall('object')
            for obj_node in object_node_list:
                lable = lable_arr.index(obj_node.find('name').text.strip())
                bndbox = obj_node.find('bndbox')
                x_min = float(bndbox.find('xmin').text.strip())
                y_min = float(bndbox.find('ymin').text.strip())
                x_max = float(bndbox.find('xmax').text.strip())
                y_max = float(bndbox.find('ymax').text.strip())
                # 位置数据用比例来表示，格式[center_x,center_y,width,height,lable]
                actual_item.append([((x_min + x_max) / 2 / img_width), ((y_min + y_max) / 2 / img_height),
                                    ((x_max - x_min) / img_width), ((y_max - y_min) / img_height), lable])
            return actual_item
        except:
            return None

    train_data = []
    actual_data = []
    file_list = random.sample(file_name_list, batch_size)
    for f_name in file_list:
        img_path = './VOC2007/JPEGImages/' + f_name
        xml_path = './VOC2007/Annotations/' + f_name.replace('.jpg', '.xml')
        if os.path.splitext(img_path)[1].lower() == '.jpg':
            actual_item = get_actual_data_from_xml(xml_path)
            img = skimage.io.imread(img_path)
            if actual_item != None:
                countwhile=0
                while True:
                    clas=[]
                    coords=[]
                    for x in actual_item:
                        clas.append(x[4])
                        coords.append([x[0],x[1],x[2],x[3]])
                    tmp0 = random.randint(-30, 50)
                    tmp1 = random.randint(-30, 50)
                    imgr=img.copy()
                    scale = np.max((400 / float(img.shape[1]),
                                    400 / float(img.shape[0])))
                    im, coords = tl.prepro.obj_box_imresize(imgr, coords,
                                                            [int(img.shape[0] * scale) + tmp0, int(img.shape[1] * scale) + tmp1],
                                                            is_rescale=True, interp='bicubic')
                    # print(im.shape)
                    # print(coords)

                    for wi in range(7):
                        imt, clast, coordst = tl.prepro.obj_box_zoom(im, clas, coords, zoom_range=(1.0, 2.2),
                                                                  fill_mode='nearest',
                                                                  order=1, is_rescale=True, is_center=True,
                                                                  is_random=True,
                                                                  thresh_wh=0.04, thresh_wh2=8.0)
                        # print(im.shape)
                        if clast!=[]:
                            im=imt
                            clas= clast
                            coords =coordst
                            break
                        if wi>=6:
                            im, clas, coords = tl.prepro.obj_box_zoom(im, clas, coords, zoom_range=(0.7, 1.2),
                                                                         fill_mode='nearest',
                                                                         order=1, is_rescale=True, is_center=True,
                                                                         is_random=True,
                                                                         thresh_wh=0.05, thresh_wh2=8.0)

                    im, coords = tl.prepro.obj_box_left_right_flip(im,
                                                                   coords, is_rescale=True, is_center=True, is_random=True)
                    # print(coords)
                    for wi in range(8):
                        imt, clast, coordst = tl.prepro.obj_box_crop(im, clas, coords,
                                                                  wrg=300, hrg=300,
                                                                  is_rescale=True, is_center=True, is_random=True,
                        thresh_wh=0.07, thresh_wh2=7.0)
                        if clast!=[]:
                            im=imt
                            clas= clast
                            coords =coordst
                            break
                        if wi==7:
                            im, clas, coords = tl.prepro.obj_box_crop(im, clas, coords,
                                                                         wrg=300, hrg=300,
                                                                         is_rescale=True, is_center=True,
                                                                         is_random=True,
                                                                         thresh_wh=0.07, thresh_wh2=8.0)


                    im = tl.prepro.illumination(im, gamma=(0.2, 1.2),
                                                contrast=(0.2, 1.2), saturation=(0.2, 1.2), is_random=True)
                    im = tl.prepro.adjust_hue(im, hout=0.1, is_offset=True,
                                              is_clip=True, is_random=True)
                    im = im / 127.5 - 1.
                    aitems = []
                    if clas!=[]:
                        for x in range(len(clas)):
                            aitem=[coords[x][0],coords[x][1],coords[x][2],coords[x][3],clas[x]]
                            aitems.append(aitem)
                        actual_data.append(aitems)
                        train_data.append(im)
                        break
                    countwhile+=1
                    if countwhile>=4:
                        clas = []
                        coords = []
                        for x in actual_item:
                            clas.append(x[4])
                            coords.append([x[0], x[1], x[2], x[3]])
                        tmp0 = random.randint(1, 30)
                        tmp1 = random.randint(1, 30)
                        imgr = img.copy()
                        im, coords = tl.prepro.obj_box_imresize(imgr, coords,
                                                                [300 + tmp0,
                                                                 300 + tmp1],
                                                                is_rescale=True, interp='bicubic')
                        im, coords = tl.prepro.obj_box_left_right_flip(im,
                                                                       coords, is_rescale=True, is_center=True,
                                                                       is_random=True)
                        im, clas, coords = tl.prepro.obj_box_crop(im, clas, coords,
                                                                     wrg=300, hrg=300,
                                                                     is_rescale=True, is_center=True,
                                                                     is_random=True,
                                                                     thresh_wh=0.02, thresh_wh2=10.0)



                        im = tl.prepro.illumination(im, gamma=(0.8, 1.2),
                                                    contrast=(0.8, 1.2), saturation=(0.8, 1.2), is_random=True)
                        im = tl.prepro.pixel_value_scale(im, 0.1, [0, 255], is_random=True)
                        im = im / 127.5 - 1.

                        aitems = []
                        if len(clas) != 0:
                            for x in range(len(clas)):
                                aitem = [coords[x][0], coords[x][1], coords[x][2], coords[x][3], clas[x]]
                                aitems.append(aitem)
                            actual_data.append(aitems)
                            train_data.append(im)
                            break
            else:
                print('Error : ' + xml_path)
                continue
    return train_data, actual_data


def jaccard(rect1, rect2):
    x_overlap = max(0, (min(rect1[0] + (rect1[2] / 2), rect2[0] + (rect2[2] / 2)) - max(rect1[0] - (rect1[2] / 2),
                                                                                        rect2[0] - (rect2[2] / 2))))
    y_overlap = max(0, (min(rect1[1] + (rect1[3] / 2), rect2[1] + (rect2[3] / 2)) - max(rect1[1] - (rect1[3] / 2),
                                                                                        rect2[1] - (rect2[3] / 2))))
    intersection = x_overlap * y_overlap
    # 删除超出图像大小的部分
    rect1_width_sub = 0
    rect1_height_sub = 0
    rect2_width_sub = 0
    rect2_height_sub = 0
    if (rect1[0] - rect1[2] / 2) < 0: rect1_width_sub += 0 - (rect1[0] - rect1[2] / 2)
    if (rect1[0] + rect1[2] / 2) > 1: rect1_width_sub += (rect1[0] + rect1[2] / 2) - 1
    if (rect1[1] - rect1[3] / 2) < 0: rect1_height_sub += 0 - (rect1[1] - rect1[3] / 2)
    if (rect1[1] + rect1[3] / 2) > 1: rect1_height_sub += (rect1[1] + rect1[3] / 2) - 1
    if (rect2[0] - rect2[2] / 2) < 0: rect2_width_sub += 0 - (rect2[0] - rect2[2] / 2)
    if (rect2[0] + rect2[2] / 2) > 1: rect2_width_sub += (rect2[0] + rect2[2] / 2) - 1
    if (rect2[1] - rect2[3] / 2) < 0: rect2_height_sub += 0 - (rect2[1] - rect2[3] / 2)
    if (rect2[1] + rect2[3] / 2) > 1: rect2_height_sub += (rect2[1] + rect2[3] / 2) - 1
    area_box_a = (rect1[2] - rect1_width_sub) * (rect1[3] - rect1_height_sub)
    area_box_b = (rect2[2] - rect2_width_sub) * (rect2[3] - rect2_height_sub)
    union = area_box_a + area_box_b - intersection
    if intersection > 0 and union > 0:
        return intersection / union,[(rect1[0]-(rect2[0]))/rect2[2],(rect1[1]-(rect2[1]))/rect2[3],math.log(rect1[2]/rect2[2]),math.log(rect1[3]/rect2[3])]

    else:
        return 0,[0.00001,0.00001,0.00001,0.00001]



def anchor_target_layers(input_actual_data,achors):
    # 生成空数组，用于保存groundtruth
    input_actual_data_len = len(input_actual_data)
    gt_class = np.zeros((input_actual_data_len, num_anchors))
    gt_location = np.zeros((input_actual_data_len, num_anchors, 4))
    gt_positives_jacc = np.zeros((input_actual_data_len, num_anchors))
    gt_positives = np.zeros((input_actual_data_len, num_anchors))
    gt_negatives = np.zeros((input_actual_data_len, num_anchors))
    background_jacc = max(0, (threshold - 0.2))
    # 初始化正例训练数据
    for img_index in range(input_actual_data_len):
        for pre_actual in input_actual_data[img_index]:
            gt_class_val = pre_actual[-1:][0]

            if gt_class_val>20 or gt_class_val<0:
                gt_class_val=0
            gt_box_val = pre_actual[:-1]
            for boxe_index in range(num_anchors):
                jacc,gt_box_val_loc = jaccard(gt_box_val, achors[boxe_index])
                if jacc > threshold or jacc == threshold:
                    gt_class[img_index][boxe_index] = gt_class_val
                    gt_location[img_index][boxe_index] = gt_box_val_loc
                    gt_positives_jacc[img_index][boxe_index] = jacc
                    gt_positives[img_index][boxe_index] = 1
                    gt_negatives[img_index][boxe_index] = 0
        # 如果没有正例，则随机创建一个正例，预防nan
        if np.sum(gt_positives[img_index]) == 0:
            # print('【没有匹配jacc】:'+str(input_actual_data[img_index]))
            random_pos_index = np.random.randint(low=0, high=num_anchors, size=1)[0]
            gt_class[img_index][random_pos_index] = 0
            gt_location[img_index][random_pos_index] = [0.00001, 0.00001, 0.00001, 0.00001]
            gt_positives_jacc[img_index][random_pos_index] = threshold
            gt_positives[img_index][random_pos_index] = 1
            gt_negatives[img_index][random_pos_index] = 0
        gt_neg_end_count = int(np.sum(gt_positives[img_index]) * 3)
        if (gt_neg_end_count + np.sum(gt_positives[img_index])) > num_anchors:
            gt_neg_end_count = num_anchors - np.sum(gt_positives[img_index])
        gt_neg_index = np.random.randint(low=0, high=num_anchors, size=gt_neg_end_count)
        for r_index in gt_neg_index:
            if gt_positives_jacc[img_index][r_index] < background_jacc and gt_positives[img_index][r_index] != 1:
                gt_class[img_index][r_index] = 0  #背景得分为0
                gt_positives[img_index][r_index] = 0
                gt_negatives[img_index][r_index] = 1

    return gt_class, gt_location, gt_positives, gt_negatives


if __name__ == '__main__':
    train_data, actual_data = get_voc(10)
    print(len(train_data),len(actual_data))
    print(train_data,actual_data)



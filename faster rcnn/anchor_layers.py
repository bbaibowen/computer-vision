
import numpy as np
import tensorflow as tf


base_size = [0,0,15,15]
anchors_ratios = [0.5, 1.0, 2.0]
anchors_scale = [8, 16, 32]
feature_map = (38,50)
base_feat = 16

def anchor():  # 个数  38*50*9 = 17100个
    '''
    先设置基础anchor，默认为[0,0,15,15],计算基础anchor的宽(16)和高(16)，anchor中心(7.5,7.5)，以及面积(256)
    计算基础anchor的面积，分别除以[0.5,1,2],得到[512,256,128]
    anchor的宽度w由三个面积的平方根值确定,得到[23,16,11]
    anchor的高度h由[23,16,11]*[0.5,1,2]确定,得到[12,16,22].
    由anchor的中心以及不同的宽和高可以得到此时的anchors.
    :param feature_map:
    :return:
    '''

    # 将默认anchor的四个坐标值转化成（宽，高，中心点横坐标，中心点纵坐标）的形式
    # 默认anchor 的值为 宽高16 中心（7.5,7.5）
    w = base_size[2] + 1  # 16
    h = base_size[3] + 1  # 16
    x = base_size[0] + 0.5 * (w - 1)  # 7.5
    y = base_size[1] + 0.5 * (h - 1)  # 7.5

    # 计算基础anchor的面积
    # anchor的宽度w由三个面积的平方根值确定, 得到[23, 16, 11]
    # anchor的高度h由[23, 16, 11] * [0.5, 1, 2] 确定, 得到[12, 16, 22].
    size = w * h
    size_ratios = size / np.array(anchors_ratios)  # [512,256,128]
    # print(size_ratios)
    ws = np.round(np.sqrt(size_ratios))
    # print(ws)
    hs = np.round(ws * anchors_ratios)

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    # print(ws)
    anchors = np.hstack((x - 0.5 * (ws - 1),
                         y - 0.5 * (hs - 1),
                         x + 0.5 * (ws - 1),
                         y + 0.5 * (hs - 1)))
    # 扩展
    all_anchors = []
    for i, an in enumerate(anchors):
        w = an[2] - an[0] + 1
        h = an[3] - an[1] + 1
        x_ctr = an[0] + 0.5 * (w - 1)
        y_ctr = an[1] + 0.5 * (h - 1)
        for j in anchors_scale:
            ws = w * j
            hs = h * j
            all_anchors.append([x_ctr - 0.5 * (ws - 1),
                                y_ctr - 0.5 * (hs - 1),
                                x_ctr + 0.5 * (ws - 1),
                                y_ctr + 0.5 * (hs - 1)])
    all_anchors = np.array(all_anchors)
    return all_anchors


def get_anchors():
    # 上述得到的是没有映射回feature map的anchors，现在要映射回feature map
    # 所以height、width是feature map的h w，对应w*h个
    # 再×16 就是每个anchors点
    # 比如生成的【1,2,3】 ----> 映射回原图就是 [16,32,48]

    height, width = feature_map
    feat_stride = base_feat

    anchors = anchor()

    x1 = tf.range(width) * feat_stride
    y1 = tf.range(height) * feat_stride
    # x1 = np.arange(0,width) * feat_stride
    # y1 = np.arange(0,height) * feat_stride
    # 向量矩阵
    x2, y2 = tf.meshgrid(x1, y1)
    # x2 , y2 = np.meshgrid(x1,y1)
    x2 = tf.reshape(x2, (-1,))
    y2 = tf.reshape(y2, (-1,))
    # 合并   拉直
    shifts = tf.transpose(tf.stack([x2, y2, x2, y2]))
    shifts = tf.transpose(
        tf.reshape(shifts, shape=[1, height * width, 4]), perm=(1, 0, 2)
    )
    anchors = anchors.reshape((1, 9, 4))
    anchors = tf.constant(anchors, tf.int32)

    all_anchors = tf.add(anchors, shifts)

    all_anchors = tf.cast(tf.reshape(all_anchors, [-1, 4]), tf.float32)

    return all_anchors



if __name__ == '__main__':
    sess = tf.Session()
    anchors = get_anchors()
    print(sess.run(anchors))
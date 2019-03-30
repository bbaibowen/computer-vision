import tensorflow as tf
import util_tf


def bboxes_intersection(bbox_ref, bboxes, name=None):
    """Compute relative intersection between a reference box and a
    collection of bounding boxes. Namely, compute the quotient between
    intersection area and box area.

    Args:
      bbox_ref: (N, 4) or (4,) Tensor with reference bounding box(es).
      bboxes: (N, 4) Tensor, collection of bounding boxes.
    Return:
      (N,) Tensor with relative intersection.

      计算参考框和

        边界框集合。也就是说，计算

        交叉区和盒子区。



        ARG:

        b框参考：（n，4）或（4，）张量与参考边界框（es）。

        bbox：（n，4）张量，边界框集合。

        返回：

        （n，）张量与相对交集。


    """
    with tf.name_scope(name, 'bboxes_intersection'):
        # Should be more efficient to first transpose.
        bboxes = tf.transpose(bboxes)
        bbox_ref = tf.transpose(bbox_ref)
        # Intersection bbox and volume.
        int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)
        # Volumes.
        inter_vol = h * w  # 各个框在[0,0,1,1]内的面积
        bboxes_vol = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])  # 各个框面积
        scores = tf.where(
            tf.greater(bboxes_vol, 0),
            tf.divide(inter_vol, bboxes_vol),
            tf.zeros_like(inter_vol),
            name='intersection')
        return scores


def bboxes_filter_overlap(labels, bboxes,
                          threshold=0.5, assign_negative=False,
                          scope=None):
    """Filter out bounding boxes based on (relative )overlap with reference
    box [0, 0, 1, 1].  Remove completely bounding boxes, or assign negative
    labels to the one outside (useful for latter processing...).

    Return:
      labels, bboxes: Filtered (or newly assigned) elements.
    """
    with tf.name_scope(scope, 'bboxes_filter', [labels, bboxes]):
        # (N,) Tensor：和[0,0,1,1]相交面积大于0的位置返回面积比（相交/原本），小于0的位置返回0
        scores = bboxes_intersection(tf.constant([0, 0, 1, 1], bboxes.dtype),
                                     bboxes)
        mask = scores > threshold
        if assign_negative:  # 保留所有的label和框，重叠区不够的label置负
            labels = tf.where(mask, labels, -labels)  # 交叉满足的标记为正，否则为负
        else:  # 删除重叠区不够的label和框
            labels = tf.boolean_mask(labels, mask)  # bool掩码，类似于array的bool切片
            bboxes = tf.boolean_mask(bboxes, mask)
        return labels, bboxes


def bboxes_resize(bbox_ref, bboxes, name=None):
    """
    使用新的参考点和基底长度（bbox_ref）重置bboxes的表示
    :param bbox_ref: 参考框，左上角点为新的参考点，hw为新的参考基
    :param bboxes: 目标框
    :param name: 域名
    :return: 目标框重新表示后的写法
    """
    # Tensors inputs.
    with tf.name_scope(name, 'bboxes_resize'):
        # Translate.
        # bbox_ref:['ymin', 'xmin', 'ymax', 'xmax']
        v = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
        bboxes = bboxes - v
        # Scale.
        s = tf.stack([bbox_ref[2] - bbox_ref[0],  # h
                      bbox_ref[3] - bbox_ref[1],  # w
                      bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1]])
        bboxes = bboxes / s
        return bboxes


def distorted_bounding_box_crop(image,
                                labels,
                                bboxes,
                                min_object_covered=0.3,
                                aspect_ratio_range=(0.9, 1.1),
                                area_range=(0.1, 1.0),
                                max_attempts=200,
                                scope=None):
    """Generates cropped_image using a one of the bboxes randomly distorted.

    See `tf.image.sample_distorted_bounding_box` for more documentation.

    Args:
        image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
        bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
            where each coordinate is [0, 1) and the coordinates are arranged
            as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
            image.
        min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
            area of the image must contain at least this fraction of any bounding box
            supplied.
        aspect_ratio_range: An optional list of `floats`. The cropped area of the
            image must have an aspect ratio = width / height within this range.
        area_range: An optional list of `floats`. The cropped area of the image
            must contain a fraction of the supplied image within in this range.
        max_attempts: An optional `int`. Number of attempts at generating a cropped
            region of the image of the specified constraints. After `max_attempts`
            failures, return the entire image.
        scope: Optional scope for name_scope.
    Returns:
        A tuple, a 3-D Tensor cropped_image and the distorted bbox
    """
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bboxes]):
        # 高级的随机裁剪
        # The bounding box coordinates are floats in `[0.0, 1.0]` relative to the width
        # and height of the underlying image.
        # 1-D, 1-D, [1, 1, 4]
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.expand_dims(bboxes, 0),  # [1, n, 4]
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,  # 最大尝试裁剪次数，失败则返回原图
            use_image_if_no_bounding_boxes=True)
        '''
        Returns:
            A tuple of `Tensor` objects (begin, size, bboxes).

        begin: A `Tensor`. Has the same type as `image_size`. 1-D, containing `[offset_height, offset_width, 0]`.
            Provide as input to `tf.slice`.
        size: A `Tensor`. Has the same type as `image_size`. 1-D, containing `[target_height, target_width, -1]`.
            Provide as input to `tf.slice`.
        bboxes: A `Tensor` of type `float32`. 3-D with shape `[1, 1, 4]` containing the distorted bounding box.
            Provide as input to `tf.image.draw_bounding_boxes`.
        '''
        # [4]，裁剪结果相对原图的(y, x, h, w)
        distort_bbox = distort_bbox[0, 0]

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        # Restore the shape since the dynamic slice loses 3rd dimension.
        cropped_image.set_shape([None, None, 3])  # <-----设置了尺寸了哈

        # Update bounding boxes: resize and filter out.
        # 以裁剪子图为参考，将bboxes更换参考点和基长度
        bboxes = bboxes_resize(distort_bbox, bboxes)  # [4], [n, 4]
        # 筛选变换后的bboxes和裁剪子图交集大于阈值的图bboxes
        labels, bboxes = bboxes_filter_overlap(labels, bboxes,
                                               threshold=0.5,
                                               assign_negative=False)
        # 返回随机裁剪的图片，筛选调整后的labels(n,)、bboxes(n, 4)，裁剪图片对应原图坐标(4,)
        return cropped_image, labels, bboxes, distort_bbox


def preprocess_image(image, labels, bboxes, out_shape,
                     scope='ssd_preprocessing_train'):

    with tf.name_scope(scope, 'ssd_preprocessing_train', [image, labels, bboxes]):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')
        # Convert to float scaled [0, 1].
        # 并不单单是float化，而是将255像素表示放缩为01表示
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # （有条件的）随机裁剪，筛选调整后的labels(n,)、bboxes(n, 4)，裁剪图片对应原图坐标(4,)
        dst_image, labels, bboxes, distort_bbox = \
            distorted_bounding_box_crop(image, labels, bboxes,
                                        min_object_covered=0.25,
                                        aspect_ratio_range=(0.6, 1.67))
        # Resize image to output size.
        dst_image = util_tf.resize_image(dst_image, out_shape,
                                         method=tf.image.ResizeMethod.BILINEAR,
                                         align_corners=False)
        
        # Randomly flip the image horizontally.
        dst_image, bboxes = util_tf.random_flip_left_right(dst_image, bboxes)

        # Randomly distort the colors. There are 4 ways to do it.
        dst_image = util_tf.apply_with_random_selector(
            dst_image,
            lambda x, ordering: util_tf.distort_color(x, ordering, False),
            num_cases=4)

        # Rescale to VGG input scale.
        image = dst_image * 255.
        image = util_tf.tf_image_whitened(image)
        # mean = tf.constant(means, dtype=image.dtype)
        # image = image - mean

        # 'NHWC' (n,) (n, 4)
        return image, labels, bboxes


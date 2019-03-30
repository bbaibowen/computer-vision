import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import numpy as np
slim = tf.contrib.slim

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

_RESIZE_SIDE_MIN = 256
_RESIZE_SIDE_MAX = 512


def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        # get_shape返回值，with_rank相当于断言assert，是否rank为指定值
        static_shape = x.get_shape().with_rank(rank).as_list()
        # tf.shape返回张量，其中num解释为"The length of the dimension `axis`."，axis默认为0
        dynamic_shape = tf.unstack(tf.shape(x), num=rank)
        # list，有定义的给数字，没有的给tensor
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]


def abs_smooth(x):  #l1平滑误差
    """Smoothed absolute function. Useful to compute an L1 smooth error.

    Define as:
        x^2 / 2         if abs(x) < 1
        abs(x) - 0.5    if abs(x) > 1
    We use here a differentiable definition using min(x) and abs(x). Clearly
    not optimal, but good enough for our purpose!
    """
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r


def reshape_list(l, shape=None):
    """Reshape list of (list): 1D to 2D or the other way around.

    Args:
      l: List or List of list.
      shape: 1D or 2D shape.
    Return
      Reshaped list.
    """
    r = []
    if shape is None:
        # Flatten everything.
        for a in l:
            if isinstance(a, (list, tuple)):
                r = r + list(a)
            else:
                r.append(a)
    else:
        # Reshape to list of list.
        i = 0
        for s in shape:
            if s == 1:
                r.append(l[i])
            else:
                r.append(l[i:i + s])
            i += s
    return r


def resize_image(image, size,
                 method=tf.image.ResizeMethod.BILINEAR,
                 align_corners=False):
    """Resize an image and bounding boxes.
    """
    # Resize image.
    with tf.name_scope('resize_image'):
        height, width, channels = tensor_shape(image)
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, size,
                                       method, align_corners)
        image = tf.reshape(image, tf.stack([size[0], size[1], channels]))
        return image


def _check3dimage(image, require_static=True):
    """Assert that we are working with properly shaped image.
    Args:
      image: 3-D Tensor of shape [height, width, channels]
        require_static: If `True`, requires that all dimensions of `image` are
        known and non-zero.
    Raises:
      ValueError: if `image.shape` is not a 3-vector.
    Returns:
      An empty list, if `image` has fully defined dimensions. Otherwise, a list
        containing an assert op is returned.
    """
    try:
        image_shape = image.get_shape().with_rank(3)
    except ValueError:
        raise ValueError("'image' must be three-dimensional.")
    if require_static and not image_shape.is_fully_defined():
        raise ValueError("'image' must be fully defined.")
    if any(x == 0 for x in image_shape):
        raise ValueError("all dims of 'image.shape' must be > 0: %s" %
                         image_shape)
    if not image_shape.is_fully_defined():
        return [tf.assert_positive(tf.shape(image),
                                          ["all dims of 'image.shape' "
                                           "must be > 0."])]
    else:
        return []


def random_flip_left_right(image, bboxes, seed=None):
    """Random flip left-right of an image and its bounding boxes.
    随机翻转图像及其边界框的左右两侧。
    """
    def flip_bboxes(bboxes):
        """Flip bounding boxes coordinates.
        """
        bboxes = tf.stack([bboxes[:, 0], 1 - bboxes[:, 3],
                           bboxes[:, 2], 1 - bboxes[:, 1]], axis=-1)
        return bboxes

    # Random flip. Tensorflow implementation.
    with tf.name_scope('random_flip_left_right'):
        image = tf.convert_to_tensor(image, name='image')
        _check3dimage(image, require_static=False)
        uniform_random = tf.random_uniform([], 0, 1.0, seed=seed)
        mirror_cond = tf.less(uniform_random, .5)
        # Flip image.
        result = tf.cond(mirror_cond,
                         lambda: tf.reverse_v2(image, [1]),
                         lambda: image)
        # Flip bboxes.
        bboxes = tf.cond(mirror_cond,
                         lambda: flip_bboxes(bboxes),
                         lambda: bboxes)

        image_shape = image.get_shape()
        result.set_shape(image_shape)
        return result, bboxes


def tf_image_whitened(image, means=(_R_MEAN, _G_MEAN, _B_MEAN)):
    """Subtracts the given means from each image channel.
    Returns:
        the centered image.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    mean = tf.constant(means, dtype=image.dtype)
    image = image - mean
    return image


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):  #扭曲颜色
    """Distort the color of a Tensor image.
    Each color distortion is non-commutative and thus ordering of the color ops
    matters. Ideally we would randomly permute the ordering of the color ops.
    Rather then adding that level of complication, we select a distinct ordering
    of color ops for each preprocessing thread.
    Args:
        image: 3-D Tensor containing single image in [0, 1].
        color_ordering: Python int, a type of distortion (valid values: 0-3).
        fast_mode: Avoids slower ops (random_hue and random_contrast)
        scope: Optional scope for name_scope.
    Returns:
        3-D Tensor color-distorted image on range [0, 1]
    Raises:
        ValueError: if color_ordering not in [0, 3]
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')
        # The random_* ops do not necessarily clamp.
        return tf.clip_by_value(image, 0.0, 1.0)


def apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1]. 随机剪裁
    Args:
        x: input Tensor.
        func: Python function to apply.
        num_cases: Python int32, number of cases to sample sel from.
    Returns:
        The result of func(x, sel), where func receives the value of the
        selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
            func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
            for case in range(num_cases)])[0]


def get_init_fn(checkpoint_path,
                train_dir,
                checkpoint_exclude_scopes,
                checkpoint_model_scope,
                model_name,
                ignore_missing_vars):
    """Returns a function run by the chief worker to warm-start the training.
    Note that the init_fn is only run when initializing the model during the very
    first global step.

    Returns:
      An init function run by the supervisor.
    """
    if checkpoint_path is None:
        return None
    # Warn the user if a checkpoint exists in the train_dir. Then ignore.
    if tf.train.latest_checkpoint(train_dir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists in %s'
            % train_dir)
        return None

    exclusions = []
    if checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in checkpoint_exclude_scopes.split(',')]
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    # Change model scope if necessary.
    if checkpoint_model_scope is not None:
        variables_to_restore = \
            {var.op.name.replace(model_name,
                                 checkpoint_model_scope): var
             for var in variables_to_restore}

    if tf.gfile.IsDirectory(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    tf.logging.info('Fine-tuning from %s. Ignoring missing vars: %s' % (checkpoint_path, ignore_missing_vars))

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=ignore_missing_vars)

if __name__ == '__main__':
    a = np.array([[2,3,4],[5,6,7]],dtype=np.float32)
    a = tf.reshape(a,[3,2])
    print(a.shape[-1])

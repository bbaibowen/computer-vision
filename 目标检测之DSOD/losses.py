import tensorflow as tf


def smooth_L1(x):
    return tf.where(tf.less_equal(tf.abs(x), 1.0), tf.multiply(0.5, tf.pow(x, 2.0)), tf.subtract(tf.abs(x), 0.5))

def loss(feature_class,feature_location,groundtruth_class,groundtruth_location,groundtruth_positives,groundtruth_count):
    softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=feature_class,
                                                                           labels=groundtruth_class)
    loss_location = tf.div(tf.reduce_sum(tf.multiply(
        tf.reduce_sum(smooth_L1(tf.subtract(groundtruth_location, feature_location)),
                      reduction_indices=2), groundtruth_positives), reduction_indices=1),
        tf.reduce_sum(groundtruth_positives, reduction_indices=1))

    # cls_loss = focal_loss(feature_class,groundtruth_class)
    cls_loss = tf.div(
        tf.reduce_sum(tf.multiply(softmax_cross_entropy, groundtruth_count), reduction_indices=1),
        tf.reduce_sum(groundtruth_count, reduction_indices=1))
    loss_all = tf.reduce_sum(tf.add(cls_loss, loss_location*5))

    return loss_all


def focal_loss(preds_cls, gt_cls,alpha=0.25, gamma=2.0):
    gt_cls = tf.one_hot(gt_cls, 21, dtype=tf.uint8)
    gt_cls = tf.to_float(gt_cls)

    preds_cls = tf.nn.sigmoid(preds_cls)
    # cross-entropy -> if y=1 : pt=p / otherwise : pt=1-p
    predictions_pt = tf.where(tf.equal(gt_cls, 1.), preds_cls, 1. - preds_cls)


    alpha_t = tf.scalar_mul(alpha, tf.ones_like(predictions_pt, dtype=tf.float32))
    alpha_t = tf.where(tf.equal(gt_cls, 1.0), alpha_t, 1.0 - alpha_t)
    gamma_t = tf.scalar_mul(gamma, tf.ones_like(predictions_pt, tf.float32))

    focal_losses = alpha_t * (-tf.pow(1.0 - predictions_pt, gamma_t) * tf.log(predictions_pt))
    # focal_losses = alpha_t * tf.pow(1. - predictions_pt, gamma) * -tf.log(predictions_pt + epsilon)
    focal_losses = tf.reduce_sum(focal_losses,1)
    return focal_losses
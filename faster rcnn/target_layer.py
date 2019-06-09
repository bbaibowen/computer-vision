import numpy as np
import numpy.random as npr
from configs import *


def bbox_overlaps(anchors, gt_boxes):
    N = anchors.shape[0]
    K = gt_boxes.shape[0]

    overlaps = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        box_area = ((gt_boxes[k, 2] - gt_boxes[k, 0] + 1) * (gt_boxes[k, 3] - gt_boxes[k, 1] + 1))
        for n in range(N):
            iw = (min(anchors[n, 2], gt_boxes[k, 2]) - max(anchors[n, 0], gt_boxes[k, 0]) + 1)
            if iw > 0:
                ih = (min(anchors[n, 3], gt_boxes[k, 3]) - max(anchors[n, 1], gt_boxes[k, 1]) + 1)
                if ih > 0:
                    ua = float(
                        (anchors[n, 2] - anchors[n, 0] + 1) * (anchors[n, 3] - anchors[n, 1] + 1) + box_area - iw * ih)
                    overlaps[n, k] = iw * ih / ua
    return overlaps


# 负责在训练RPN的时候，从上万个anchor中选择一些(比如256)进行训练，以使得正负样本比例大概是1:1. 同时给出训练的位置参数目标
def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, _feat_stride, all_anchors, num_anchor_types):
    """Same as the anchor target layer in original Fast/er RCNN """
    A = num_anchor_types  # anchor类型数量，为9
    num_of_total_anchors = all_anchors.shape[0]  # 所有anchor数量
    K = num_of_total_anchors / num_anchor_types  # feature_map点数

    # rpn_cls_score.shape=[1,height,width,depth],depth为18,height与width分别为原图高/16,原图宽/16
    height, width = rpn_cls_score.shape[1:3]

    # only keep anchors inside the image
    # 只保存图像区域内的anchor，超出图片区域的舍弃
    # im_info[0]存的是图片像素行数即高，im_info[1]存的是图片像素列数即宽
    # [0]表示,np.where取出的是tuple，里面是一个array，array里是符合的引索，所以[0]就是要取出array索引
    index_inside = np.where(
        (all_anchors[:, 0] >= -0) & (all_anchors[:, 1] >= -0) &
        (all_anchors[:, 2] < im_info[1]) & (all_anchors[:, 3] < im_info[0]))[0]

    # keep only inside anchors
    anchors = all_anchors[index_inside, :]

    # label: 1 is positive, 0 is negative, -1 is dont care
    # 生成一个具有符合条件的anchor数个数的未初始化随机数的ndarray
    labels = np.empty((len(index_inside),), dtype=np.float32)
    # 将这些随机数初始化为-1
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    # 此时假设通过筛选的anchor的个数为N，GT个数为K
    # 产生一个(N,K)array，此K与上面说的K不同.里面每一项存的是第N个anchor相对于第K个GT的IOU（重叠面积/（anchor+GT-重叠面积））
    overlaps = bbox_overlaps(np.ascontiguousarray(anchors, dtype=np.float),  # TODO:!!!
                             np.ascontiguousarray(gt_boxes, dtype=np.float))
    # 每个anchor对应的最大
    argmax_overlaps = overlaps.argmax(axis=1)  # 位置
    max_overlaps = overlaps[np.arange(len(index_inside)), argmax_overlaps]  # iou值

    # 每个GT对应的最大
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])]

    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]  # TODO:看不懂->每个GT对应的最大anchors的位置（第几个）

    # assign bg labels first so that positive labels can clobber them
    # first set the negatives
    # 将max_overlaps（与lables大小相同，其实都是对应与anchor）小于0.3的都认为是bg（back ground），设置标签为0
    labels[max_overlaps < 0.3] = 0

    # fg label: for each gt, anchor with highest overlap
    # 与gt有最佳匹配的anchor，labels设置为1（gt_argmax_overlaps虽然与labels形状不同，但是gt_argmax_overlaps存的是anchor的index，就对该index的anchor进行赋值）
    # 多个gt可能有同一个最佳匹配的anchor，此时lebals的该anchor引索位置被重复赋值为1
    labels[gt_argmax_overlaps] = 1

    # fg label: above threshold IOU
    # 与gt重叠参数大于等于0.7的anchor，labels设置为1
    labels[max_overlaps >= 0.7] = 1

    # subsample positive labels if we have too many
    # 如果我们有太多前景样本，减少前景样本
    num_fg = int(0.5 * 256)  # TODO：256是总数，0.5是比例，提取
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    # 如果我们有太多背景样本，减少背景样本
    num_bg = 256 - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1

    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])  # 前一个参数：anchors，后一个参数：与每个anchorIOU最大的GT
    # bbox_targets为每个anchor变为最相近GT的回归参数

    # only the positive ones have regression targets
    # bbox_inside_weights 它实际上就是控制回归的对象的，只有真正是前景的对象才会被回归
    # 对应labels==1的引索,全零的四个元素变为(1.0, 1.0, 1.0, 1.0)
    bbox_inside_weights = np.zeros((len(index_inside), 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = np.array((1.0, 1.0, 1.0, 1.0))  # TODO:

    # bbox_outside_weights 就是为了平衡 box_loss,cls_loss 的，因为二个loss差距过大，所以它被设置为 1/N 的权重
    bbox_outside_weights = np.zeros((len(index_inside), 4), dtype=np.float32)
    num_examples = np.sum(labels >= 0)
    positive_weights = np.ones((1, 4)) * 1.0 / num_examples
    negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    bbox_outside_weights[labels == 1, :] = positive_weights  # 1/256
    bbox_outside_weights[labels == 0, :] = negative_weights

    # map up to original set of anchors
    # 之后可能还会用到第一次被筛选出的anchor信息，所以对labels信息进行扩充，添加进去了第一次筛选出的anchor的标签（都为-1）
    labels = _unmap(labels, num_of_total_anchors, index_inside, fill=-1)
    # 以下三个相同，都是把原始anchor信息添加进去，但是信息都是0
    bbox_targets = _unmap(bbox_targets, num_of_total_anchors, index_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, num_of_total_anchors, index_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, num_of_total_anchors, index_inside, fill=0)

    # labels
    labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, 1, A * height, width))
    rpn_labels = labels

    # bbox_targets
    bbox_targets = bbox_targets.reshape((1, height, width, A * 4))
    rpn_bbox_targets = bbox_targets

    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A * 4))
    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, A * 4))
    rpn_bbox_outside_weights = bbox_outside_weights

    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (experiments) back to the original set of items (of size count) """
    ## 判断label是否为一维的
    if len(data.shape) == 1:
        ## 建立一个（anchor总数量，）大小的一维数组
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ## 图片内的anchor属于第一次筛选，筛选出去的label都为-1
        ## 第一次筛选后的anchor，其中符合条件的anchor分别被赋予0与1，其余的都为-1
        ## 第二次筛选：可能标签为1与0的太多了，随机排除一些，标签设置为-1
        ## 所以inds_inside与labels一一对应，但是其中还存在有大量不训练的标签为-1的anchor
        ret[inds] = data
    else:
        ## 产生一个（anchor总数量，4）ndarray
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ## 对于标签为0与1的填入信息
        ret[inds, :] = data
    return ret


def bbox_transform(ex_rois, gt_rois):
    # (左上x,左上y,右下x,右下y)
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ## 中心点坐标
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)
    # 纵向堆叠后转置
    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    # 返回的List,每一行为一个(targets_dx, targets_dy, targets_dw, targets_dh)
    return targets


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""
    # 要求anchor与对应匹配最好GT个数相同
    assert ex_rois.shape[0] == gt_rois.shape[0]
    # 要有anchor左上角与右下角坐标，有4个元素
    assert ex_rois.shape[1] == 4
    # GT有标签位，所以为5个
    assert gt_rois.shape[1] == 5  # TODO:原始是5
    # 返回一个用于anchor回归成target的包含每个anchor回归值(dx、dy、dw、dh)的array
    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)


# 负责在训练RoIHead/Fast R-CNN的时候，从RoIs选择一部分(比如128个)用以训练
def proposal_target_layer(rpn_rois, rpn_scores, gt_boxes, _num_classes):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
    # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
    all_rois = rpn_rois
    all_scores = rpn_scores

    rois_per_image = BATCH_SIZE
    fg_rois_per_image = np.round(FG_FRACTION * rois_per_image)

    # Sample rois with classification labels and bounding box regression
    # targets
    labels, rois, roi_scores, bbox_targets, bbox_inside_weights = _sample_rois(
        all_rois, all_scores, gt_boxes, fg_rois_per_image,
        rois_per_image, _num_classes)

    rois = rois.reshape(-1, 5)
    roi_scores = roi_scores.reshape(-1)
    labels = labels.reshape(-1, 1)
    bbox_targets = bbox_targets.reshape(-1, _num_classes * 4)
    bbox_inside_weights = bbox_inside_weights.reshape(-1, _num_classes * 4)
    bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)

    return rois, roi_scores, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


#
#
def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]  # 去除背景
    for ind in inds:
        cls = clss[ind]
        start = int(4 * cls)
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = (1.0, 1.0, 1.0, 1.0)
    return bbox_targets, bbox_inside_weights


#
#
def _compute_targets_with_labels(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)

    return np.hstack((labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


#
#
def _sample_rois(all_rois, all_scores, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    # overlaps: (rois x gt_boxes)
    overlaps = bbox_overlaps(
        np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
        np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
    gt_assignment = overlaps.argmax(axis=1)  # roi与第几个GT重叠最多
    max_overlaps = overlaps.max(axis=1)  # 最大IOU为
    labels = gt_boxes[gt_assignment, 4]

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where(max_overlaps < FG_THRESH)[0]  # TODO：参看原代码

    # Small modification to the original version where we ensure a fixed number of regions are sampled
    # 总共选择128个区域，如果前景背景都有，那先选前景，数量多余要求的数时选择要求的个数，少于时则有几个选几个。剩下的都用背景补齐128
    if fg_inds.size > 0 and bg_inds.size > 0:
        fg_rois_per_image = min(fg_rois_per_image, fg_inds.size)
        fg_inds = npr.choice(fg_inds, size=int(fg_rois_per_image), replace=False)
        bg_rois_per_image = rois_per_image - fg_rois_per_image
        to_replace = bg_inds.size < bg_rois_per_image  # 如果背景数量少于要求的数量，则可以重复选择
        bg_inds = npr.choice(bg_inds, size=int(bg_rois_per_image), replace=to_replace)
    # 没有背景时，不选背景，128个都为前景，可以重复
    elif fg_inds.size > 0:
        to_replace = fg_inds.size < rois_per_image
        fg_inds = npr.choice(fg_inds, size=int(rois_per_image), replace=to_replace)
        fg_rois_per_image = rois_per_image
    elif bg_inds.size > 0:
        # 没有前景时，选128个可重复的背景
        to_replace = bg_inds.size < rois_per_image
        bg_inds = npr.choice(bg_inds, size=int(rois_per_image), replace=to_replace)
        fg_rois_per_image = 0
    else:
        import pdb
        pdb.set_trace()

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Select sampled values from various arrays:
    labels = labels[keep_inds]
    # Clamp labels for the background RoIs to 0
    labels[int(fg_rois_per_image):] = 0
    rois = all_rois[keep_inds]
    roi_scores = all_scores[keep_inds]
    # 每个类别对应的回归值
    bbox_target_data_with_label = _compute_targets_with_labels(rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4],
                                                               labels)

    bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(bbox_target_data_with_label, num_classes)

    return labels, rois, roi_scores, bbox_targets, bbox_inside_weights

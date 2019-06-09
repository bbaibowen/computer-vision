import numpy as np
import torch
from bounding_box import BoxList
PRE_NMS_THRESH = 0.05
PRE_NMS_TOK_K = 1000
NMS_THRESH = 0.6
FPN_POS_NMS_TOP_K = 100
CLASS = 100


def select_layers(loc,cls,box,centerness,img_size):

    batch,channel,height,width = cls.shape
    cls = cls.permute(0,2,3,1)
    cls = cls.reshape(batch,-1,channel).sigmoid()
    box = box.permute(0,2,3,1)
    box = box.reshape(batch,-1,4)
    centerness = centerness.permute(0,2,3,1)
    centerness = centerness.reshape(batch,-1).sigmoid()

    select_id = cls > PRE_NMS_THRESH
    num_id = select_id.view(batch,-1).sum(1)
    num_id = num_id.clamp(max = PRE_NMS_TOK_K)

    cls = cls * centerness[:,:,None]

    res = []

    for i in range(batch):
        cls_i = cls[i]
        select_id_i = select_id[i]
        cls_i = cls_i[select_id_i]
        per_candidate_nonzeros = select_id_i.nonzero()
        loc_i = per_candidate_nonzeros[:,0]
        per_class = per_candidate_nonzeros[:,1] + 1

        box_i = box[i]
        box_i = box_i[loc_i]
        per_locs = loc[loc_i]

        tok_k = num_id[i]

        if select_id_i.sum().item() > tok_k.item():
            cls_i,top_k_index = cls_i.topk(tok_k,sorted = False)
            per_class = per_class[top_k_index]
            box_i = box_i[top_k_index]
            per_locs = per_locs[top_k_index]

        l_ = per_locs[:,0] - box_i[:,0]
        t_ = per_locs[:,1] - box_i[:,1]
        r_ = per_locs[:,0] + box_i[:,2]
        b_ = per_locs[:,1] + box_i[:,3]

        regs = torch.stack([l_,t_,r_,b_],dim=1)
        h,w = img_size[i]
        boxlist = BoxList(regs,(int(w),int(h)),mode = 'xyxy')
        boxlist.add_field("labels", per_class)
        boxlist.add_field("scores", cls_i)
        boxlist = boxlist.clip_to_image(remove_empty=False)
        # boxlist = remove_small_boxes(boxlist, self.min_size)
        res.append(boxlist)

    return res

def _cat(tensors, dim=0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def cat_boxlist(bboxes):
    """
    Concatenates a list of BoxList (having the same image size) into a
    single BoxList

    Arguments:
        bboxes (list[BoxList])
    """
    assert isinstance(bboxes, (list, tuple))
    assert all(isinstance(bbox, BoxList) for bbox in bboxes)

    size = bboxes[0].size
    assert all(bbox.size == size for bbox in bboxes)

    mode = bboxes[0].mode
    assert all(bbox.mode == mode for bbox in bboxes)

    fields = set(bboxes[0].fields())
    assert all(set(bbox.fields()) == fields for bbox in bboxes)

    cat_boxes = BoxList(_cat([bbox.bbox for bbox in bboxes], dim=0), size, mode)

    for field in fields:
        data = _cat([bbox.get_field(field) for bbox in bboxes], dim=0)
        cat_boxes.add_field(field, data)

    return cat_boxes


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.

    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    def _box_nms(boxes,score,nms_thresh):

        x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = score.argsort()
        print(order)
        print(order.shape)
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i],x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (area[i] + area[order[1:]] - inter)  # 计算IOU

            inds = np.where(ovr <= nms_thresh)[0]
            order = order[inds + 1]
        return keep
    if nms_thresh <= 0:
        return boxlist
    mode = boxlist.mode
    boxlist = boxlist.convert("xyxy")
    boxes = boxlist.bbox
    score = boxlist.get_field(score_field)
    keep = _box_nms(boxes, score, nms_thresh)
    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist.convert(mode)


def select_over_all_levels(boxlist):
    num_images = len(boxlist)
    results = []

    for i in range(num_images):
        scores = boxlist[i].get_field("scores")
        labels = boxlist[i].get_field("labels")
        boxes = boxlist[i].bbox
        boxlist = boxlist[i]
        result = []

        for j in range(1, CLASS):
            inds = (labels == j).nonzero().view(-1)
            scores_j = scores[inds]
            boxes_j = boxes[inds, :].view(-1, 4)
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class = boxlist_nms(
                boxlist_for_class, NMS_THRESH,
                score_field="scores"
            )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j,
                                     dtype=torch.int64,
                                     device=scores.device)
            )
            result.append(boxlist_for_class)
        result = cat_boxlist(result)
        number_of_detections = len(result)

        if number_of_detections > FPN_POS_NMS_TOP_K > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(),
                number_of_detections - FPN_POS_NMS_TOP_K + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        results.append(result)
        return results

#对五个scale的feature map进行筛选
def _select(locs,cls,box,centerness,img_size):

    sample_boxes = []
    for _,(l,o,b,c) in enumerate(zip(locs,cls,box,centerness)):
        sample_boxes.append(select_layers(l,o,b,c,img_size))

    boxlist = list(zip(*sample_boxes))
    boxlist = [cat_boxlist(i) for i in boxlist]
    boxlist = select_over_all_levels(boxlist)
    return boxlist


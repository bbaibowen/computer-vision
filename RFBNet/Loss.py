import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max() # 求x中的最大值
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max



def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """
        idx: (int) current batch index，因为每次都是batch size图像输入训练的

    """

    def encode(matched, priors, variances):

        g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
        g_cxcy /= (variances[0] * priors[:, 2:])
        g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
        g_wh = torch.log(g_wh) / variances[1]

        return torch.cat([g_cxcy, g_wh], 1)

    #(cx, cy, w, h) -> (x1, y1, x2, y2)
    def point_form(boxes):

        return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                          boxes[:, :2] + boxes[:, 2:] / 2), 1)

    def intersect(box_a, box_b):

        A = box_a.size(0)
        B = box_b.size(0)
        max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                           box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                           box_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

    def jaccard(box_a, box_b):

        inter = intersect(box_a, box_b)
        area_a = ((box_a[:, 2] - box_a[:, 0]) *
                  (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, 2] - box_b[:, 0]) *
                  (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
        union = area_a + area_b - inter  # A ∪ B
        return inter / union



    overlaps = jaccard(
        truths,
        point_form(priors)
    )

    # [1,num_objects] best prior for each ground truth，
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior，
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)

    #gt bbox匹配max IoU的anchor，
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    # 分配gt bbox与label
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx]          # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background，
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior
    return conf_t,loc_t



class MultiBoxLoss(nn.Module):

    def __init__(self):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = 21
        self.threshold = 0.5 # IoU阈值
        self.background_label = 0
        self.negpos_ratio = 3 # 正负样本比例，1:3
        self.variance = [0.1,0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data = predictions # (batch_size,num_priors,4) / (batch_size,num_priors,21)
        priors = priors # (num_priors,4)
        num = loc_data.size(0) # batch_size
        num_priors = (priors.size(0)) # num_priors

        #批次cls和box 【batch,...】
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)

        for idx in range(num):
            truths = targets[idx][:,:-1].data #gt box
            labels = targets[idx][:,-1].data # label
            defaults = priors
            # 这里没有返回东西，但是直接传入了loc_t,conf_t，而可以看到他是按批次进行match的，而且传入了idx，
            # 意思就是把总的传入match，在match中为当前批次进行了赋值
            # 即con_i,box_i = match() ,conf_t[i,...] = con_i,loc_t[i,...] = box_i。一个意思
            match(self.threshold,truths,defaults,self.variance,labels,loc_t,conf_t,idx)


        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t,requires_grad=False)

        #此时的conf_t全是类别对应的index，所以就是不要label是bg的
        pos = conf_t > 0

        # Localization Loss (Smooth L1)
        # loc Shape: [batch,num_priors,4]

        #pos.unsqueeze(pos.dim()):拓展了最后一个维度，相当于 np.expand_dims(,axis = 2)-->[batch,num_anchor,1]
        #expand_as:拓展成像括号里面的参数的维度一样，这里为什么不可以直接expand_as，因为pos原来就两个维度，你得先拓展成三维，
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data) # 相当于取出所有正样本对应的index位置,[batch,num_anchor,4]


        #这里先计算全部的，后面进行OHEM
        loc_p = loc_data[pos_idx].view(-1,4)
        loc_t = loc_t[pos_idx].view(-1,4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False) # Localization Loss (Smooth L1)
        #这个就是box loss，OHEM对的只是分类loss


        batch_conf = conf_data.view(-1,self.num_classes)#[batch * num_anchor,21]

        #log_sum_exp(batch_conf) --->[batch * num_anchor,1]
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1,1)) #正负样本分类loss
        # OHEM
        loss_c[pos.view(-1,1)] = 0 # filter out pos boxes for now，OHEM操作不考虑正样本，仅在负样本上操作
        loss_c = loss_c.view(num, -1) # 按图像归类各个负样本
        _,loss_idx = loss_c.sort(1, descending=True) # loss降序排序，那么仅需要选择前面的高loss即可
        _,idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1,keepdim=True)  # 正样本数量
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1) # 由正样本数量按1：3比例得出需保留的难负样本数量
        neg = idx_rank < num_neg.expand_as(idx_rank) # 结合_,idx_rank = loss_idx.sort(1)理解，为了取出难neg pred bbox

        # Confidence Loss Including Positive and Negative Examples最终只有难负样本loss + 正样本loss参与模型参数的更新
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes) # pred bbox的预测结果，经过难负样本挖掘后留存下来的
        targets_weighted = conf_t[(pos+neg).gt(0)] # 剩余需要计算cls gt label，包含了正负样本的gt label
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False) # 分类的交叉熵损失函数

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = max(num_pos.data.sum().float(), 1) # N: number of matched default boxes
        loss_l /= N
        loss_c /= N
        return loss_l + loss_c

import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import tranpose_and_gather_feat

pull_weight=0.1
push_weight=0.1
regr_weight=1.

class Loss(nn.Module):

    def __init__(self):
        super(Loss,self).__init__()

    def focal(self,heat,heat_gt):
        pos = heat_gt.eq(1)
        neg = heat_gt.lt(1)
        focal_loss = 0
        for i,j in enumerate(heat):

            neg_weights = torch.pow(1 - heat_gt[i][neg[i]],4)
            pos_pred = j[pos[i] == 1.]
            neg_pred = j[neg[i] == 1.]

            pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred,2)
            neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred,2) * neg_weights
            num_pos = pos[i].float().sum()
            pos_loss = pos_loss.sum()
            neg_loss = neg_loss.sum()
            if pos_pred.nelement() == 0:
                focal_loss -= neg_loss

            else:
                focal_loss -= (pos_loss + neg_loss) / num_pos
        return focal_loss

    def offset_L1(self,offset,gt,mask):
        num = mask.float().sum() * 2
        mask = mask.unsqueeze(2).expand_as(gt)
        offset = offset[mask == 1]
        gt = gt[mask == 1]
        loss = F.smooth_l1_loss(offset,gt,size_average=False)
        loss = loss / num

        return loss

    def ae_loss(self, tag0, tag1, masks):
        num = masks.sum(dim=1, keepdim=True).unsqueeze(1).expand_as(tag0)

        masks = masks.unsqueeze(2)
        tag_mean = (tag0 + tag1) / 2
        tag0 = torch.pow(tag0 - tag_mean, 2) / (num + 1e-4)
        tag0 = (tag0 * masks).sum()
        tag1 = torch.pow(tag1 - tag_mean, 2) / (num + 1e-4)
        tag1 = (tag1 * masks).sum()
        pull = tag0 + tag1
        mask = masks.unsqueeze(1) + masks.unsqueeze(2)
        mask = mask.eq(2)
        num = num.unsqueeze(2).expand_as(mask)

        num2 = (num - 1) * num
        m = 2

        dist = tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2)
        dist = m - torch.abs(dist)
        dist = nn.functional.relu(dist, inplace=True)
        dist = dist - m / (num + 1e-4)
        dist = dist / (num2 + 1e-4)
        dist = dist[mask]
        push = dist.sum()
        return pull, push


    def forward(self, outs,targets):
        tl_heat, br_heat, tl_tag, br_tag, tl_regr, br_regr = outs
        tl_heat_gt, br_heat_gt, tl_tag_gt, br_tag_gt, tl_regr_gt, br_regr_gt,tag_mask = targets

        #heat
        tl_heat = tl_heat.sigmoid()
        br_heat = br_heat.sigmoid()
        tl_heat_loss = self.focal(tl_heat,tl_heat_gt)
        br_heat_loss = self.focal(br_heat,br_heat_gt)
        heat_loss = tl_heat_loss + br_heat_loss

        #offset smooth l1
        tl_regr = tranpose_and_gather_feat(tl_regr,tl_tag)
        br_regr = tranpose_and_gather_feat(br_regr,br_tag)
        tl_offset_loss = self.offset_L1(tl_regr,tl_regr_gt,tag_mask) * regr_weight
        br_offset_loss = self.offset_L1(br_regr,br_regr_gt,tag_mask) * regr_weight
        offset_loss = tl_offset_loss + br_offset_loss

        #emb
        tl_tag = tranpose_and_gather_feat(tl_tag,tl_tag_gt)
        br_tag = tranpose_and_gather_feat(br_tag,br_tag_gt)
        pull_loss, push_loss = self.ae_loss(tl_tag,br_tag,tag_mask)

        total_loss = heat_loss + offset_loss + pull_loss + push_loss
        total_loss /= len(tl_heat)

        return total_loss



if __name__ == '__main__':
    pass







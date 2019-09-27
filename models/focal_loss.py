import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def _expand_binary(self, pred, target):
        if target.dim() == 2:
            onehot = torch.zeros(pred.size()).cuda()
            onehot.scatter_(1, target.unsqueeze(-1), 1.0)
            return onehot
        else:
            return target.type_as(pred)

    def forward(self, pred, target):
        pred_sigmoid = pred.sigmoid()
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * 
                       (1 - target)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none') * focal_weight
        return loss.sum(dim=1).mean()
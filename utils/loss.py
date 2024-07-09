from einops import rearrange

import torch 
import torch.nn as nn 
import torch.nn.functional as F


def cal_loss(pred, gt, dice_loss_model, focal_loss_model, dice_weight:float=0.8):
    gt = gt.squeeze(1).to(torch.long)
    focal_loss = focal_loss_model(pred, gt)
    dice_loss = dice_loss_model(pred, gt)
    
    loss = (1 - dice_weight) * focal_loss + dice_weight * dice_loss

    return loss
    

def set_one_hot(pred, gt):
    _, num_cls, _, _ = pred.shape
    pred = torch.softmax(pred, dim=1)
    gt = F.one_hot(gt.squeeze(1).to(torch.long), num_cls).permute(0,3,1,2)

    return pred, gt


# Segment Loss funcation
# https://github.com/OpenGVLab/SAM-Med2D/blob/main/utils.py
class FocalLoss(nn.Module):
    def __init__(self, gamma=3.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, mask):
        """
        pred: [B, C, H, W]
        mask: [B, C, H, W]
        """
        pred, mask = set_one_hot(pred, mask)
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        focal_loss = 0.0
        
        _, num_cls, _, _ = mask.shape
        for i in range(num_cls):
            p, m = pred[:, i], mask[:, i]
            num_pos = torch.sum(m)
            num_neg = m.numel() - num_pos
            w_pos = (1 - p) ** self.gamma
            w_neg = p ** self.gamma

            loss_pos = -self.alpha * m * w_pos * torch.log(p + 1e-12)
            loss_neg = -(1 - self.alpha) * (1 - m) * w_neg * torch.log(1 - p + 1e-12)

            loss = (torch.sum(loss_pos) + torch.sum(loss_neg)) / (num_pos + num_neg + 1e-12)
            focal_loss = focal_loss + loss

        focal_loss = focal_loss / num_cls
        return focal_loss


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-3):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, mask):
        """
        pred: [B, C, H, W]
        mask: [B, C, H, W]
        """
        pred, mask = set_one_hot(pred, mask)
        assert pred.shape == mask.shape, "pred and mask should have the same shape."
        dice_loss = 0.0
        
        _, num_cls, _, _ = mask.shape
        for i in range(num_cls):
            p, m = pred[:, i], mask[:, i]
            intersection = torch.sum(p * m)
            union = torch.sum(p*p) + torch.sum(m*m)
            loss = 1 - (2 * intersection + self.smooth) / (union + self.smooth)
            dice_loss = dice_loss + loss
        
        dice_loss = dice_loss / num_cls
        return dice_loss


class MaskIoULoss(nn.Module):

    def __init__(self, ):
        super(MaskIoULoss, self).__init__()

    def forward(self, pred_mask, ground_truth_mask, pred_iou):
        """
        pred_mask: [B, 1, H, W]
        ground_truth_mask: [B, 1, H, W]
        pred_iou: [B, 1]
        """
        assert pred_mask.shape == ground_truth_mask.shape, "pred_mask and ground_truth_mask should have the same shape."

        p = torch.sigmoid(pred_mask)
        intersection = torch.sum(p * ground_truth_mask)
        union = torch.sum(p) + torch.sum(ground_truth_mask) - intersection
        iou = (intersection + 1e-7) / (union + 1e-7)
        iou_loss = torch.mean((iou - pred_iou) ** 2)
        return iou_loss


class FocalDiceloss_IoULoss(nn.Module):
    
    def __init__(self, weight=20.0, iou_scale=1.0):
        super(FocalDiceloss_IoULoss, self).__init__()
        self.weight = weight
        self.iou_scale = iou_scale
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.maskiou_loss = MaskIoULoss()

    def forward(self, pred, mask, pred_iou):
        """
        pred: [B, 1, H, W]
        mask: [B, 1, H, W]
        """
        assert pred.shape == mask.shape, "pred and mask should have the same shape."

        focal_loss = self.focal_loss(pred, mask)
        dice_loss =self.dice_loss(pred, mask)
        loss1 = self.weight * focal_loss + dice_loss
        loss2 = self.maskiou_loss(pred, mask, pred_iou)
        loss = loss1 + loss2 * self.iou_scale
        return loss


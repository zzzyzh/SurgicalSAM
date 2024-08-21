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
    

# Segment Loss funcation
class DiceLoss(nn.Module):
    def __init__(self, ignore_index=None, smooth=1e-5):
        """
        Initialize DiceLoss.
        :param ignore_index: Class index to ignore, e.g., the background class. None means no class is ignored.
        :param smooth: Smoothing factor to prevent division by zero.
        """
        super(DiceLoss, self).__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, pred, mask):
        """
        Compute the Dice Loss.
        :param pred: Model predictions, shape [B, C, H, W]
        :param mask: Ground truth labels, shape [B, H, W]
        :param softmax: Whether to apply softmax to the predictions
        """
        
        pred = F.softmax(pred, dim=1)
        num_classes = pred.shape[1]
        mask = F.one_hot(mask, num_classes).permute(0, 3, 1, 2).float()  # Convert to one-hot encoding
        
        if self.ignore_index is not None:
            mask = mask[:, 1:, ...]  # Ignore the class specified by self.ignore_index
            pred = pred[:, 1:, ...]

        intersection = torch.sum(pred * mask, dim=(0, 2, 3))
        union = torch.sum(pred * pred, dim=(0, 2, 3)) + torch.sum(mask * mask, dim=(0, 2, 3))
        dice_score = 2.0 * intersection / (union + self.smooth)
        
        loss = 1 - dice_score
        return loss.mean()  # Return the average Dice Loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=9):
        super(FocalLoss, self).__init__()
        assert 0 <= alpha < 1, "alpha should be in [0, 1)"
        self.alpha = torch.tensor([alpha] + [1 - alpha] * (num_classes - 1))
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds, labels):
        """
        Calculate focal loss for segmentation
        :param preds: predictions from the model, shape: [B, C, H, W]
        :param labels: ground truth labels, shape: [B, H, W]
        :return: computed focal loss
        """
        if not preds.shape[1] == self.num_classes:
            raise ValueError(f"Expected input tensor to have {self.num_classes} channels, got {preds.shape[1]}")
        
        preds = preds.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        preds = preds.view(-1, self.num_classes)  # Flatten [N, C] where N is B*H*W
        labels = labels.view(-1)  # Flatten label tensor
        
        preds_logsoft = F.log_softmax(preds, dim=1)  # Log softmax on the class dimension
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.unsqueeze(1)).squeeze(1)
        preds_logsoft = preds_logsoft.gather(1, labels.unsqueeze(1)).squeeze(1)
        
        self.alpha = self.alpha.to(preds.device)  # Ensure alpha is on the correct device
        alpha = self.alpha.gather(0, labels)
        
        loss = -alpha * torch.pow((1 - preds_softmax), self.gamma) * preds_logsoft
        return loss.mean()


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


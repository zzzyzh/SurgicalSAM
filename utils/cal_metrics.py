import numpy as np 
import torch 


def eval_metrics(eval_masks, gt_eval_masks, num_classes=4):
    """Given the predicted masks and groundtruth annotations, predict the challenge IoU, IoU, mean class IoU, and the IoU for each class
        
      ** The evaluation code is taken from the official evaluation code of paper: ISINet: An Instance-Based Approach for Surgical Instrument Segmentation
      ** at https://github.com/BCV-Uniandes/ISINet
      
    Args:
        eval_masks (dict): the dictionary containing the predicted mask for each frame 
        gt_eval_masks (dict): the dictionary containing the groundtruth mask for each frame 

    Returns:
        dict: a dictionary containing the evaluation results for different metrics 
    """

    dice_results = dict()
    iou_results = dict()
    
    all_im_dice_acc = []
    all_im_iou_acc = []
    all_im_iou_acc_challenge = []
    
    im_dices = {c: [] for c in range(1, num_classes+1)}
    im_ious = {c: [] for c in range(1, num_classes+1)}    
    class_dices = {c: [] for c in range(1, num_classes+1)}
    class_ious = {c: [] for c in range(1, num_classes+1)}
    cum_I, cum_U = 0, 0
    
    for file_name, prediction in eval_masks.items():
        
        full_mask = gt_eval_masks[file_name]
        im_dice = []
        im_iou = []
        im_iou_challenge = []
        target = full_mask.numpy()
        gt_classes = np.unique(target)
        gt_classes.sort()
        gt_classes = gt_classes[gt_classes > 0] 
        if np.sum(prediction) == 0:
            if target.sum() > 0: 
                all_im_dice_acc.append(0)
                all_im_iou_acc.append(0)
                all_im_iou_acc_challenge.append(0)
                for class_id in gt_classes:
                    im_dices[class_id].append(0)
                    im_ious[class_id].append(0)
                    class_dices[class_id].append(0)
                    class_ious[class_id].append(0)
            continue

        gt_classes = torch.unique(full_mask)
        # loop through all classes from 1 to num_classes 
        for class_id in range(1, num_classes+1): 

            current_pred = (prediction == class_id).astype(np.float64)
            current_target = (full_mask.numpy() == class_id).astype(np.float64)

            if current_pred.astype(np.float64).sum() != 0 or current_target.astype(np.float64).sum() != 0:
                dice = compute_mask_dice(current_pred, current_target)
                im_dice.append(dice)
                i, u = compute_mask_IU_eval(current_pred, current_target)     
                im_iou.append(i/u)
                cum_I += i
                cum_U += u
                im_dices[class_id].append(dice)
                im_ious[class_id].append(i/u)
                class_dices[class_id].append(dice)
                class_ious[class_id].append(i/u)
                if class_id in gt_classes:
                    im_iou_challenge.append(i/u)
        
        if len(im_dice) > 0:
            all_im_dice_acc.append(np.nanmean(im_dice))
        if len(im_iou) > 0:
            all_im_iou_acc.append(np.nanmean(im_iou))
        if len(im_iou_challenge) > 0:
            all_im_iou_acc_challenge.append(np.nanmean(im_iou_challenge))

    # calculate final metrics
    mean_im_dice = np.nanmean(all_im_dice_acc)
    mean_im_iou = np.nanmean(all_im_iou_acc)
    mean_im_iou_challenge = np.nanmean(all_im_iou_acc_challenge)
    mean_class_dice = torch.tensor([torch.tensor(values).float().mean() for c, values in class_dices.items() if len(values) > 0]).mean().item()
    mean_class_iou = torch.tensor([torch.tensor(values).float().mean() for c, values in class_ious.items() if len(values) > 0]).mean().item()
    mean_imm_iou = cum_I / (cum_U + 1e-15)
    
    final_im_dice = torch.zeros(9)
    final_class_im_iou = torch.zeros(9)
    dice_per_class = []
    cIoU_per_class = []
    for c in range(1, num_classes + 1):
        final_im_dice[c-1] = torch.tensor(im_dices[c]).float().mean()
        dice_per_class.append(round((final_im_dice[c-1]*100).item(), 2))
        final_class_im_iou[c-1] = torch.tensor(class_ious[c]).float().mean()
        cIoU_per_class.append(round((final_class_im_iou[c-1]*100).item(), 2))
        
    iou_results["challengIoU"] = round(mean_im_iou_challenge*100,2)
    iou_results["IoU"] = round(mean_im_iou*100,2)
    iou_results["mcIoU"] = round(mean_class_iou*100,2)
    iou_results["mIoU"] = round(mean_imm_iou*100,2)
    iou_results["cIoU_per_class"] = cIoU_per_class
    
    dice_results["Dice"] = round(mean_im_dice*100,2)
    dice_results["mcDice"] = round(mean_class_dice*100,2)
    dice_results["Dice_per_class"] = dice_per_class
    
    return iou_results, dice_results


def compute_mask_dice(masks, target):
    """compute dice used for evaluation
    """
    assert target.shape[-2:] == masks.shape[-2:]
    intersection = (masks * target).sum()
    union = (masks + target).sum()
    dice = (2. * intersection) / union
    return dice


def compute_mask_IU_eval(masks, target):
    """compute iou used for evaluation
    """
    assert target.shape[-2:] == masks.shape[-2:]
    temp = masks * target
    intersection = temp.sum()
    union = ((masks + target) - temp).sum()
    return intersection, union

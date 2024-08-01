import os
import os.path as osp 
import random 
import argparse
from tqdm import tqdm
import csv

import numpy as np 
import torch 
from torch.utils.data import DataLoader
from torch.nn import functional as F

from dataset import TestingDataset
from model import Few_Shot_SAM
from utils.utils import create_masks, read_gt_masks, get_logger, vis_pred
from utils.cal_metrics import eval_metrics, compute_hd95


def get_args():
    print('======> Process Arguments')
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default='prototype_sam')
    
    # Set-up Model
    parser.add_argument('--task', type=str, default='ven', help='specify task')
    parser.add_argument('--dataset', type=str, default='bhx_sammed', help='specify dataset')
    parser.add_argument('--data_root_dir', type=str, default='../../data', help='specify dataset root path')
    parser.add_argument('--save_dir', type=str, default='../../data/experiments', help='specify save path')
    parser.add_argument('--num_classes', type=int, default=4, help='specify the classes of the dataset without the bg')
    parser.add_argument('--sam_mode', type=str, default='vit_b', choices=['vit_b', 'vit_l'], help='specify backbone')
    parser.add_argument('--train_time', type=str, default=None, help='specify the training time')
    parser.add_argument('--model_type', type=str, default='lora', 
                        choices=[
                            'lora',
                            'wo_encoder'
                        ],
                        help='specify the parameters involved in training')

    # Testing Strategy
    parser.add_argument('--scale', type=float, default=0.1, help='percentage of training data')
    parser.add_argument('--cls_id', type=int, default=0, help='the id of the class to segment')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=512, help='image_size')
    parser.add_argument('--resolution', type=int, default=512, help='image_size')
    parser.add_argument('--thr', type=float, default=0.5, help='threshold of the pred')
    parser.add_argument('--num_tokens', type=int, default=4, help='the num of prompts')
    parser.add_argument('--volume', type=bool, default=False, help='whether to evaluate test set in volume')
    parser.add_argument('--vis', type=bool, default=False, help='whether to visualise results')

    args = parser.parse_args()
    return args

def test(args):
    print('======> Set Parameters for Inference' )  
    run_name = args.run_name
        
    seed = 2024  
    batch_size = args.batch_size
    num_workers = args.num_workers

    thr = args.thr
    volume = args.volume # 是否将 2D slice 合成 3D volume 后再推理
    vis = args.vis

    # set seed for reproducibility 
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    
    print('======> Load Dataset-Specific Parameters' )
    scale = args.scale
    sam_mode = args.sam_mode
    model_type = args.model_type
    dataset_name = args.dataset
    image_size = args.image_size
    resolution = args.resolution
    
    data_root_dir = osp.join(args.data_root_dir, args.task, args.dataset)
    cls = args.cls_id
    test_dataset = TestingDataset(
                        data_root_dir=data_root_dir,
                        mode='test',
                        image_size=image_size
                        )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    print('======> Set Saving Directories and Logs')
    now = args.train_time
    task = f'{sam_mode}_{model_type}_{now}'
    settings = f'few_shot_{int(scale*100)}'
    save_dir = osp.join(args.save_dir, run_name, dataset_name, settings, task) 
    save_log_dir = osp.join(save_dir, 'log')
    save_ckpt_dir = osp.join(save_dir, 'ckpt')
    save_pred_dir = osp.join(save_dir, 'pred', 'test')
    os.makedirs(save_pred_dir, exist_ok=True) 
    
    if not volume:
        loggers = get_logger(os.path.join(save_log_dir, f'{task}_test_{cls}.log'))
    else:
        loggers = get_logger(os.path.join(save_log_dir, f'{task}_test_volume_{cls}.log'))
    loggers.info(f'Args: {args}')
    
    print('======> Load Prototype-based Model for different model mode')
    sam_checkpoint = osp.join(save_ckpt_dir, f'{task}_ckpt.pth') 
    num_classes = args.num_classes
    feat_size = resolution // 16

    model = Few_Shot_SAM(
                sam_checkpoint=sam_checkpoint,
                sam_mode=sam_mode,   
                model_type=model_type,
                mask_size=image_size,
                feat_size=feat_size,
                num_classes=num_classes,
                resolution=resolution,
            ) 
          
    model.cuda()
    model.load_state_dict(torch.load(sam_checkpoint))

    print('======> Start Inference')
    val_masks = dict()
    model.eval()

    with torch.no_grad():
        tbar = tqdm((test_dataloader), total = len(test_dataloader), leave=False)
            
        for batch_input in tbar:   
            mask_names = batch_input['mask_names']
            masks = batch_input['masks'].cuda() 
            images = batch_input['images'].cuda()
            images = F.interpolate(images, (resolution, resolution), mode='bilinear', align_corners=False)
            
            outputs, _, _, _ = model(images, masks, mode='test')
            preds = outputs['preds'] 
            preds = torch.argmax(torch.softmax(preds, dim=1), dim=1).squeeze(0) # [b, 512, 512]

            for pred, im_name in zip(preds, mask_names):
                val_masks[im_name] = np.array(pred.detach().cpu())
        
    gt_masks = read_gt_masks(data_root_dir=data_root_dir, mode='test', cls_id=cls, mask_size=image_size, volume=volume)
    val_masks = create_masks(data_root_dir, val_masks, image_size, mode='test', volume=volume)
    iou_results, dice_results, iou_csv, dice_csv = eval_metrics(val_masks, gt_masks, num_classes)
    loggers.info(f'IoU_Results: {iou_results};')
    loggers.info(f'Dice_Results: {dice_results}.')
    with open(os.path.join(save_log_dir, 'results_volume.csv' if volume else 'results.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(iou_csv)
        writer.writerow(dice_csv)
        
    if volume:
        metric_hd95 = []
        for i in range(1, num_classes+1):
            metric_cls_hd95 = []
            for key in val_masks.keys():   
                metric_cls_hd95.append(compute_hd95(val_masks[key]==i, gt_masks[key]==i))
            metric_hd95.append(np.mean(metric_cls_hd95, axis=0))
        hd95 = np.mean(metric_hd95, axis=0)
        loggers.info(f'HD95: {round(hd95, 2)}.')
    
    if vis and not volume:
        vis_pred(val_masks, gt_masks, save_pred_dir, num_classes)


if __name__ == '__main__':
    args = get_args()
    test(args)
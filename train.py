import os
import os.path as osp 
import random 
import argparse
from datetime import datetime
from tqdm import tqdm
from einops import rearrange

import numpy as np 
import torch 
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from dataset import TrainingDataset, TestingDataset
from model import Few_Shot_SAM, cal_cls_embedding
from utils.utils import read_gt_masks, get_logger, vis_pred
from utils.loss import DiceLoss
from pytorch_metric_learning import losses
from utils.cal_metrics import eval_metrics


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
    parser.add_argument('--sam_ckpt', type=str, default='../../data/experiments/weights/sam_vit_b_01ec64.pth', help='specify raw SAM ckpt path')
    parser.add_argument('--model_type', type=str, default='lora', 
                        choices=[
                            'lora',
                            'wo_encoder'
                        ],
                        help='specify the parameters involved in training')
    parser.add_argument('--rank', type=int, default=5, help='Rank for LoRA adaptation')

    # Training Strategy
    parser.add_argument('--scale', type=float, default=0.1, help='percentage of training data')
    parser.add_argument('--num_epochs', type=int, default=300, help='the num of epochs')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--image_size', type=int, default=512, help='image size')
    parser.add_argument('--resolution', type=int, default=512, choices=[256, 512], help='input size of the model')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'adamw'], help='')
    parser.add_argument('--dice_weight', type=float, default=0.9)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--num_tokens', type=int, default=4, help='the num of prompts') 
    parser.add_argument('--vis', type=bool, default=False, help='whether to visualise results')
    
    parser.add_argument('--debug', type=bool, default=False, help='whether to use debug mode')


    args = parser.parse_args()
    return args


def train(args):
    print('======> Set Parameters for Training' )  
    run_name = args.run_name
        
    seed = 2024  
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    num_workers = args.num_workers

    vis = args.vis

    # set seed for reproducibility 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print('======> Load Prototype-based Model for different model mode')
    sam_checkpoint = args.sam_ckpt
    sam_mode = args.sam_mode
    model_type = args.model_type
    num_classes = args.num_classes
    image_size = args.image_size
    resolution = args.resolution
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

    # set requires_grad to False to the whole model 
    for params in model.parameters():
        params.requires_grad=False
        
    # finetune correct weights
    for name, params in model.named_parameters(): 
        if model_type != 'wo_encoder':
            if 'image_encoder' in name and 'lora' in name:
                params.requires_grad = True
        if 'mask_decoder' in name:
            params.requires_grad = True
        if 'prototype' in name:
            params.requires_grad = True

    model.cuda()

    print('======> Load Dataset-Specific Parameters')
    scale = args.scale
    dataset_name = args.dataset
    data_root_dir = osp.join(args.data_root_dir, args.task, args.dataset)
    train_dataset = TrainingDataset(
                        data_root_dir = data_root_dir,
                    )
    val_dataset = TestingDataset(
                        data_root_dir = data_root_dir,
                        mode='val',
                    )       
    
    def worker_init_fn(worker_id):
        random.seed(seed + worker_id)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, worker_init_fn=worker_init_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=16, worker_init_fn=worker_init_fn)
    gt_masks = read_gt_masks(data_root_dir=data_root_dir, mask_size=image_size, mode='val')
    
    print('======> Set Saving Directories and Logs')
    now = datetime.now().strftime('%Y%m%d-%H%M')
    task = f'{sam_mode}_{model_type}_{now}'
    if args.debug:
        save_dir = osp.join(args.save_dir, 'debug', model_type)
    else:
        save_dir = osp.join(args.save_dir, run_name, dataset_name, f'few_shot_{int(scale*100)}', task)          
    writer = SummaryWriter(osp.join(save_dir, 'runs'))             
    save_log_dir = osp.join(save_dir, 'log')
    save_ckpt_dir = osp.join(save_dir, 'ckpt')
    save_pred_dir = osp.join(save_dir, 'pred', 'val')
    os.makedirs(save_ckpt_dir, exist_ok=True) 
    os.makedirs(save_log_dir, exist_ok=True) 
    os.makedirs(save_pred_dir, exist_ok=True) 
     
    loggers = get_logger(os.path.join(save_log_dir, f'{task}.log'))
    loggers.info(f'Args: {args}')  
    # for name, params in model.named_parameters(): 
    #     if params.requires_grad:
    #         loggers.info(name) 
                    
    print('======> Define Optmiser and Loss')
    dice_loss_model = DiceLoss().cuda()
    # focal_loss_model = FocalLoss().cuda()
    contrastive_loss_model = losses.NTXentLoss(temperature=0.07).cuda()
    
    lr = args.lr
    weight_decay = args.weight_decay
    optimizer_schedule = args.optimizer
    
    # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    if optimizer_schedule == 'adam':
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)  
    elif optimizer_schedule == 'sgd':
        optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    elif optimizer_schedule == 'adamw':
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999))
    
    model_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_total_params = sum(p.numel() for p in model.parameters())
    loggers.info('model_grad_params:' + str(model_grad_params))
    loggers.info('model_total_params:' + str(model_total_params))
    
    print('======> Start Training and Validation' )
    best_dice_val = -100.0
    
    for epoch in range(num_epochs):   
        
        # training 
        model.train()
        train_loss, train_seg_loss, train_contrastive_loss = 0, 0, 0
        tbar = tqdm((train_dataloader), total = len(train_dataloader), leave=False)

        for batch_input in tbar: 
            masks = batch_input['masks'].cuda() 
            images = batch_input['images'].cuda()
            images = F.interpolate(images, (resolution, resolution), mode='bilinear', align_corners=False)  
            
            outputs, prototypes, sam_feats, feat_list, cls_ids = model(images, masks, mode='train')
            class_embeddings = cal_cls_embedding(sam_feats, masks, feat_list, feat_size)
            
            # compute loss     
            seg_loss = dice_loss_model(outputs['preds'], masks.squeeze(1).to(torch.long))
            contrastive_loss = contrastive_loss_model(prototypes, torch.tensor([i for i in range(1, prototypes.size()[0] + 1)]).cuda(), ref_emb=class_embeddings, ref_labels=cls_ids)
            loss = seg_loss + contrastive_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.detach().cpu()
            train_seg_loss += seg_loss.detach().cpu()
            train_contrastive_loss += contrastive_loss.detach().cpu()
            
            tbar.set_description(f'Train Epoch [{epoch+1}/{num_epochs}]')
            tbar.set_postfix(loss = loss.item())
            
        loggers.info(f'Train - Epoch: {epoch+1}/{num_epochs}; Average Train Loss: {train_loss/len(train_dataloader)}')
        writer.add_scalar(tag='train/loss', scalar_value=train_loss/len(train_dataloader), global_step=epoch)
        writer.add_scalar(tag='train/seg_loss', scalar_value=train_seg_loss/len(train_dataloader), global_step=epoch)
        writer.add_scalar(tag='train/contrastive_loss', scalar_value=train_contrastive_loss/len(train_dataloader), global_step=epoch)
        
        # validation 
        val_masks = dict()
        model.eval()
        
        with torch.no_grad():
            
            val_seg_loss = 0
            vbar = tqdm((val_dataloader), total = len(val_dataloader), leave=False)
            
            for batch_input in vbar:    
                mask_names = batch_input['mask_names']
                masks = batch_input['masks'].cuda() 
                images = batch_input['images'].cuda()
                images = F.interpolate(images, (resolution, resolution), mode='bilinear', align_corners=False)
                
                outputs, _, _, _, _ = model(images, masks, mode='test')
                preds = outputs['preds'] 
                preds = torch.argmax(torch.softmax(preds, dim=1), dim=1).squeeze(0) # [b, 512, 512]

                for pred, im_name in zip(preds, mask_names):
                    val_masks[im_name] = np.array(pred.detach().cpu())
            
                # compute loss                 
                val_seg_loss = dice_loss_model(outputs['preds'], masks.squeeze(1).to(torch.long))

                vbar.set_description(f'Val Epoch [{epoch+1}/{num_epochs}]')

        loggers.info(f'Validation - Epoch: {epoch+1}/{num_epochs}; Average Val Loss: {val_seg_loss/len(val_dataloader)}')

        iou_results, dice_results, _, _ = eval_metrics(val_masks, gt_masks, num_classes)
                
        loggers.info(f'Validation - Epoch: {epoch+1}/{num_epochs};')   
        loggers.info(f'IoU_Results: {iou_results};')
        loggers.info(f'Dice_Results: {dice_results}.')
        writer.add_scalar(tag='val/iou', scalar_value=iou_results['IoU'], global_step=epoch)
        writer.add_scalar(tag='val/dice', scalar_value=dice_results['Dice'], global_step=epoch)     
        
        if vis and (epoch+1)%(num_epochs//4) == 0:
            save_epoch_dir = osp.join(save_pred_dir, f'epoch_{epoch+1}')
            os.makedirs(save_epoch_dir, exist_ok=True)
            vis_pred(val_masks, gt_masks, save_epoch_dir, num_classes)
        
        if dice_results['Dice'] > best_dice_val:
            best_dice_val = dice_results['Dice']
            torch.save(model.state_dict(), osp.join(save_ckpt_dir, f'{task}_ckpt.pth'))

            loggers.info(f'Best Dice: {best_dice_val:.4f} at Epoch {epoch+1}')        

    writer.close()      
    

if __name__ == '__main__':
    args = get_args()
    train(args)
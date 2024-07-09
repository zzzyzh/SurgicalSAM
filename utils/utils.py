import os 
import os.path as osp 
import logging
from tqdm import tqdm

from PIL import Image
import SimpleITK as sitk
import cv2 
import numpy as np 
import torch 
import matplotlib.pyplot as plt


# sabs
PART_TEST_VOLUME = ["0029", "0003", "0001", "0004", "0025", "0035"] # reference: https://github.com/Beckschen/TransUNet


def save_array_as_nii_volume(data, spacing_raw):
    euler3d = sitk.Euler3DTransform()
    img = sitk.GetImageFromArray(data)
    img.SetSpacing((spacing_raw[2], spacing_raw[3], spacing_raw[1]))
    xsize, ysize, zsize = img.GetSize()
    xspacing, yspacing, zspacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()

    spacing = [1, 1, 1.2]

    new_size = (
        int(xsize * xspacing / spacing[0]), int(ysize * yspacing / spacing[1]), int(zsize * zspacing / spacing[2]))
    img = sitk.Resample(img, new_size, euler3d, sitk.sitkNearestNeighbor, origin, spacing, direction)

    return img


def read_gt_masks(data_root_dir="/home/yanzhonghao/data/ven/bhx_sammed", mode="val", cls_id=0, mask_size=512, volume=False):   
    """Read the annotation masks into a dictionary to be used as ground truth in evaluation.

    Returns:
        dict: mask names as key and annotation masks as value 
    """
    gt_eval_masks = dict()
    
    gt_eval_masks_path = osp.join(data_root_dir, mode, "masks")
    if not volume:
        for mask_name in sorted(os.listdir(gt_eval_masks_path)):
            mask = cv2.imread(osp.join(gt_eval_masks_path, mask_name), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (mask_size, mask_size), interpolation=cv2.INTER_NEAREST)
            if cls_id != 0:
                mask[mask != cls_id] = 0
            mask = torch.from_numpy(mask)
            gt_eval_masks[mask_name] = mask
    else:
        if 'bhx' in data_root_dir:
            spacing_list = np.loadtxt(f'{data_root_dir}/{mode}/spacing.txt', delimiter=',', dtype=float).tolist()
            for _, spacing in enumerate(tqdm(spacing_list)):
                id = str(int(spacing[0])).zfill(4)
                mask_as_png = np.zeros([40, mask_size, mask_size], dtype='uint8')
                for mask_name in sorted(os.listdir(gt_eval_masks_path)):
                    if f'{id}_' in mask_name:
                        mask = cv2.imread(osp.join(gt_eval_masks_path, mask_name), 0)
                        mask = cv2.resize(mask, (mask_size, mask_size), interpolation=cv2.INTER_NEAREST)
                        if cls_id != 0:
                            mask[mask != cls_id] = 0
                        mask = Image.fromarray(mask)
                        i = int(mask_name.split('.')[0].split('_')[-1])
                        mask_as_png[i, :, :] = mask
                np.transpose(mask_as_png, [2, 0, 1])
                mask_as_nii = save_array_as_nii_volume(mask_as_png, spacing)
                gt_eval_masks[id] = torch.tensor(sitk.GetArrayFromImage(mask_as_nii))
        elif 'sabs' in data_root_dir:
            for id in tqdm(PART_TEST_VOLUME):
                mask_as_png = np.zeros([200, mask_size, mask_size], dtype='uint8')
                for mask_name in sorted(os.listdir(gt_eval_masks_path)):
                    if f'{id}_' in mask_name:
                        mask = cv2.imread(osp.join(gt_eval_masks_path, mask_name), 0)
                        mask = cv2.resize(mask, (mask_size, mask_size), interpolation=cv2.INTER_NEAREST)
                        if cls_id != 0:
                            mask[mask != cls_id] = 0
                        i = int(mask_name.split('.')[0].split('_')[-1])
                        mask_as_png[i, :, :] = mask

                gt_eval_masks[id] = torch.tensor(mask_as_png)
                
    return gt_eval_masks
  

def create_masks(data_root_dir, val_masks, mask_size, mode='test', volume=False):
    if not volume:
        eval_masks = val_masks
    else:
        eval_masks = dict()
        if 'bhx' in data_root_dir:
            spacing_list = np.loadtxt(f'{data_root_dir}/{mode}/spacing.txt', delimiter=',', dtype=float).tolist()
            for _, spacing in enumerate(tqdm(spacing_list)):
                id = str(int(spacing[0])).zfill(4)
                mask_as_png = np.zeros([40, mask_size, mask_size], dtype='uint8')
                for mask_name, mask in val_masks.items():
                    if id in mask_name:
                        i = int(mask_name.split('.')[0].split('_')[-1])
                        mask_as_png[i, :, :] = mask.astype(np.uint8)
                np.transpose(mask_as_png, [2, 0, 1])
                mask_as_nii = save_array_as_nii_volume(mask_as_png, spacing)
                eval_masks[id] = sitk.GetArrayFromImage(mask_as_nii)  
        elif 'sabs' in data_root_dir:
            for id in tqdm(PART_TEST_VOLUME):
                mask_as_png = np.zeros([200, mask_size, mask_size], dtype='uint8')
                for mask_name, mask in val_masks.items():
                    if id in mask_name:
                        i = int(mask_name.split('.')[0].split('_')[-1])
                        mask_as_png[i, :, :] = mask.astype(np.uint8)
                eval_masks[id] = mask_as_png      
             
    return eval_masks

        
def get_logger(filename, write_mode="w", verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fh = logging.FileHandler(filename, write_mode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def vis_pred(pred_dict, gt_dict, save_dir, num_classes):    
    
    color_mapping = {
        1: (11, 158, 150),   
        2: (27, 0, 255),     
        3: (255, 0, 255),     
        4: (241, 156, 118),    
        5: (27, 255, 255),    
        6: (227, 0, 127),    
        7: (255, 255, 0),    
        8: (0, 255, 0)     
    }

    for _, mask_name in enumerate(tqdm(list(pred_dict.keys()))):
        pred = np.array(pred_dict[mask_name])  # [256, 256]
        gt = np.array(gt_dict[mask_name])
        
        # 初始化 RGB 图像
        pred_vis = np.zeros((*pred.shape, 3), dtype=np.uint8)
        gt_vis = np.zeros((*gt.shape, 3), dtype=np.uint8)
        
        # 应用颜色映射
        for cls, color in color_mapping.items():
            pred_mask = (pred == cls)
            gt_mask = (gt == cls)
            
            pred_vis[pred_mask] = color
            gt_vis[gt_mask] = color

        plt.figure(figsize=(6, 3)) # 设置画布大小
        plt.subplot(1, 2, 1)  # 1行2列的子图中的第1个
        plt.imshow(gt_vis)  # 使用灰度颜色映射
        plt.title('Ground Truth')  # 设置标题
        plt.axis('off')  # 关闭坐标轴
        
        plt.subplot(1, 2, 2)  # 1行2列的子图中的第2个
        plt.imshow(pred_vis)  # 使用灰度颜色映射
        plt.title('Prediction')  # 设置标题
        plt.axis('off')  # 关闭坐标轴
        
        plt.savefig(os.path.join(save_dir, mask_name))
        plt.close()



        
if __name__ == '__main__':
    read_gt_masks()
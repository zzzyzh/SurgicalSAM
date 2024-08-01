import os 
import os.path as osp 
import logging
from tqdm import tqdm

import cv2 
import numpy as np 
import torch 
import matplotlib.pyplot as plt


# bhx
BHX_TEST_VOLUME = [str(i).zfill(4) for i in range(701, 801)]
# sabs
SABS_TEST_VOLUME = ["0018", "0002", "0000", "0003", "0014", "0024"]


def read_gt_masks(data_root_dir="../../data/ven/bhx_sammed", mode="val", cls_id=0, mask_size=512, volume=False):   
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
            test_volume = BHX_TEST_VOLUME
            test_len = 40
        elif 'sabs' in data_root_dir:
            test_volume = SABS_TEST_VOLUME
            test_len = 200
        
        for id in tqdm(test_volume):
            mask_as_png = np.zeros([test_len, mask_size, mask_size], dtype='uint8')
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
            test_volume = BHX_TEST_VOLUME
            test_len = 40
        elif 'sabs' in data_root_dir:
            test_volume = SABS_TEST_VOLUME
            test_len = 200

        for id in tqdm(test_volume):
            mask_as_png = np.zeros([test_len, mask_size, mask_size], dtype='uint8')
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


CLASS2COLOR = {
    0: (0, 0, 0),
    1: (255, 124, 116),
    2: (223, 120, 242),
    3: (0, 176, 0),
    4: (0, 108, 240),
    5: (239, 209, 207),
    6: (167, 245, 155),
    7: (255, 113, 165),
    8: (255, 172, 168),
    9: (235, 168, 233),
    10: (209, 209, 209),
}


def vis_pred(pred_dict, gt_dict, save_dir, dataset_name):    
    
    if dataset_name == 'bhx_sammed':
        color_mapping = {i: CLASS2COLOR[i] for i in range(1, 5)}
    elif dataset_name == 'sabs_sammed' or 'sabs_sammed_roi':
        color_mapping = {i: CLASS2COLOR[i] for i in range(1, 9)}

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
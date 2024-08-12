import os 
import os.path as osp
import re
import random 
import math
import cv2 
from PIL import Image
from tqdm import tqdm
from einops import rearrange, repeat

import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as F
from scipy import ndimage
from scipy.ndimage import zoom
from scipy.ndimage import gaussian_filter, map_coordinates


BHX_TRAIN_LIST = [str(i).zfill(4) for i in range(600)]
SABS_TRAIN_LIST = ["0004", "0005", "0006", "0008", "0009", "0010", "0012", "0013", "0015", "0016", "0017", "0019", "0020", "0022", "0023", "0026", "0028", "0029"]


class TrainingDataset(Dataset):
    def __init__(self, 
                data_root_dir = "/home/yanzhonghao/data/ven/bhx_sammed", 
                image_size = 512,
                scale = 0.1,
                ):
        
        self.image_size = image_size       

        # directory
        self.dataset = data_root_dir.split('/')[-1]
        self.image_dir = osp.join(data_root_dir, 'train', "images")
        self.mask_dir = osp.join(data_root_dir, 'train', "masks")
        
        self.support = self.get_support(scale)
        
        # normalization
        self.pixel_mean=[123.675, 116.28, 103.53]
        self.pixel_std=[58.395, 57.12, 57.375]
        
    def __len__(self):
        return len(self.support)

    def __getitem__(self, index):
        im_name = self.support[index]
        
        image_path = osp.join(self.image_dir, im_name)
        mask_path = osp.join(self.mask_dir, im_name)
        
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        image = Image.fromarray(image)
        mask = Image.fromarray(mask)
        
        image, mask = self.augmentation(image, mask, rotate_angle=30, colour_factor=0.4)
        image, mask = np.asarray(image), np.asarray(mask)
                
        image = torch.tensor(image).to(torch.float32)
        image = (image - torch.tensor(self.pixel_mean).view(-1, 1, 1)) / torch.tensor(self.pixel_std).view(-1, 1, 1)
        mask = torch.tensor(mask).unsqueeze(0).to(torch.float32)

        batch_input = {
            'images': image,
            'masks': mask,
        }
        
        return batch_input

    def get_support(self, scale):
        if 'bhx' in self.dataset:
            support_idx = random.sample(BHX_TRAIN_LIST, int(len(BHX_TRAIN_LIST)*scale)+1)
        elif 'sabs' in self.dataset:
            support_idx = random.sample(SABS_TRAIN_LIST, int(len(SABS_TRAIN_LIST)*scale)+2)
        
        pat_list = sorted(os.listdir(self.mask_dir))
        pat_idx = [im_name for im_name in pat_list if im_name.split('_')[1] in support_idx]        
        slice_idx = random.sample(pat_idx, math.ceil(len(pat_list)*scale))

        return sorted(slice_idx)
    
    def augmentation(
            self, 
            image, 
            mask, 
            rotate_angle,
            colour_factor
            ):
        """Generate augmentation to image and masks
        image - original image
        mask - binary mask for the classes present in the image

        Returns:
            image - image after the augmentation
            mask - mask after the augmentation 
        """
        
        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Rotate 
        if random.random() > 0.5:
            angle = rotate_angle * random.random() * (random.random()>0.5)
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
            
        return image, mask
        

class TestingDataset(Dataset):
    def __init__(self, 
                 data_root_dir = "/home/yanzhonghao/data/ven/bhx_sammed", 
                 mode = "test", 
                 image_size = 512,
                 ):
        
        self.image_size = image_size       

        # directory
        self.dataset = data_root_dir.split('/')[-1]
        self.image_dir = osp.join(data_root_dir, mode, "images")
        self.mask_dir = osp.join(data_root_dir, mode, "masks")
        self.mask_list = sorted(os.listdir(self.mask_dir))
                
        # normalization
        self.pixel_mean=[123.675, 116.28, 103.53]
        self.pixel_std=[58.395, 57.12, 57.375]
        
    def __len__(self):
        return len(self.mask_list)

    def __getitem__(self, index):
        mask_name = self.mask_list[index]
        
        image_path = osp.join(self.image_dir, mask_name)
        mask_path = osp.join(self.mask_dir, mask_name)
        
        image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = torch.tensor(image).to(torch.float32)
        image = (image - torch.tensor(self.pixel_mean).view(-1, 1, 1)) / torch.tensor(self.pixel_std).view(-1, 1, 1)
        mask = torch.tensor(mask).unsqueeze(0).to(torch.float32)

        batch_input = {
            'images': image,
            'masks': mask,
            'mask_names': mask_name
        }

        return batch_input
   

if __name__ == '__main__':
    random.seed(2024)
    
    data_root_dir = '../../data/ven/bhx_sammed'
    # data_root_dir = '../../data/abdomen/sabs_sammed'
    
    train_dataset = TrainingDataset(
                                data_root_dir = data_root_dir,
                                )
    # train_dataset = TestingDataset(
    #                     data_root_dir = data_root_dir,
    #                     cls_id=0,
    #                     mode="test",
    #                     image_size=256
    #                     )
    
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=16)
    
    for epoch in range(30):
        tbar = tqdm((train_dataloader), total = len(train_dataloader), leave=False)
        for batch_input in tbar:
            masks = batch_input['masks']
            print(torch.max(masks))

        
    


    
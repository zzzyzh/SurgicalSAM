import random
from einops import rearrange

from PIL import Image
import cv2
import numpy as np
import torch 
import torch.nn as nn 
from torch.nn import functional as F

from segment_anything import sam_model_registry


class Few_Shot_SAM(nn.Module):
    def __init__(
        self,
        sam_checkpoint,
        sam_mode,
        model_type,
        mask_size = 512,
        feat_size = 32, 
        num_classes = 4,
        resolution = 512,
        ):

        super().__init__()
        print("======> Load SAM" )
        self.num_classes = num_classes
        self.sam_mode = sam_mode
        self.model_type = model_type
        self.feat_size = feat_size
        self.image_encoder, self.prompt_encoder, self.mask_decoder = sam_model_registry[sam_mode](checkpoint=sam_checkpoint, model_type=model_type, image_size=resolution, num_classes=num_classes)             
        
        print("======> Load Prototypes and Prototype-based Prompt Encoder" )
        self.feat_size = feat_size
        self.learnable_prototypes_model = Learnable_Prototypes(feat_dim = 256, num_classes = num_classes)
        self.prototype_prompt_encoder =  Prototype_Prompt_Encoder(feat_dim = 256, 
                                                                  feat_size = feat_size,
                                                                  hidden_dim_dense = 128, 
                                                                  hidden_dim_sparse = 128, 
                                                                  num_tokens = 4,
                                                                  num_classes = num_classes)  
        
        with open(sam_checkpoint, "rb") as f:
            state_dict = torch.load(f)
            if 'model' in state_dict.keys():
                state_dict = state_dict['model']
            sam_pn_embeddings_weight = {k.split("prompt_encoder.point_embeddings.")[-1]: v for k, v in state_dict.items() if k.startswith("prompt_encoder.point_embeddings") and ("0" in k or "1" in k)}
            sam_pn_embeddings_weight_ckp = {"0.weight": torch.concat([sam_pn_embeddings_weight['0.weight'] for _ in range(4)], dim=0),
                                            "1.weight": torch.concat([sam_pn_embeddings_weight['1.weight'] for _ in range(4)], dim=0)}
            missing, unexpected = self.prototype_prompt_encoder.pn_cls_embeddings.load_state_dict(sam_pn_embeddings_weight_ckp, strict=False)  
        
        self.resolution = resolution            
        self.mask_size = (mask_size, mask_size)
            
    def forward(self, images, gts, mode='train'):
        prototypes = self.learnable_prototypes_model() # [cls, 256]

        image_embeddings = self.image_encoder(images) # [b, 256, 32, 32]
        image_embeddings = rearrange(image_embeddings, 'b c h w -> b (h w) c')
        
        cls_ids = []
        for gt in gts:
            unique = torch.unique(gt)
            cls_ids.append(unique[1:])
        
        sam_feats = []
        for i in range(len(cls_ids)):
            image_embedding = torch.stack([image_embeddings[i] for _ in range(len(cls_ids[i]))], dim=0)
            sam_feats.append(image_embedding)
        
        feat_list = [np.array(cls_id.detach().cpu()).astype(np.uint8) for cls_id in cls_ids]
        cls_ids = torch.concat(cls_ids, dim=0).to(torch.long) # [b']
        sam_feats = torch.concat(sam_feats, dim=0) # [b', 1024, 256]
        cls_embeddings = cal_cls_embedding(sam_feats, gts, feat_list, self.feat_size)
        
        sam_feats, dense_embeddings, sparse_embeddings = self.prototype_prompt_encoder(sam_feats, prototypes, cls_ids)
        
        low_res_masks_list, iou_predictions_list = [], []
        for dense_embedding, sparse_embedding, features_per_image in zip(dense_embeddings.unsqueeze(1), sparse_embeddings.unsqueeze(1), sam_feats):
            low_res_mask, iou_prediction = self.mask_decoder(
                image_embeddings=features_per_image.unsqueeze(0), # [1, 256, feat_size, feat_size]
                image_pe=self.prompt_encoder.get_dense_pe(),  # [1, 256, feat_size, feat_size]
                sparse_prompt_embeddings=sparse_embedding, # [1, num_cls*num_tokens, 256]
                dense_prompt_embeddings=dense_embedding,  # [1, 256, feat_size, feat_size]
                multimask_output=False,
            )
            low_res_masks_list.append(low_res_mask)
            iou_predictions_list.append(iou_prediction)
            
        low_res_masks_list = torch.concat(low_res_masks_list, dim=0)
        iou_predictions_list = torch.concat(iou_predictions_list, dim=0)

        i = 0
        low_res_masks, iou_predictions = [], []
        for cls_id in feat_list:
            mask_slice = slice(i, i+len(cls_id))
            low_res_mask = low_res_masks_list[mask_slice].squeeze(1)
            iou_prediction = iou_predictions_list[mask_slice]

            mask = torch.zeros(self.num_classes+1, low_res_mask.shape[1], low_res_mask.shape[2])
            iou_pred = torch.zeros(self.num_classes+1)

            for j in range(len(cls_id)):
                mask[cls_id[j]] = low_res_mask[j]
                iou_pred[cls_id[j]] = iou_prediction[j]
                
            low_res_masks.append(mask.unsqueeze(0)) # [num_cls+1, 128, 128]
            iou_predictions.append(iou_pred.unsqueeze(0)) # [num_cls+1]

            i += len(cls_id)
        
        low_res_masks = torch.concat(low_res_masks, dim=0).cuda() # [b, num_cls+1, 128, 128]
        iou_predictions = torch.concat(iou_predictions, dim=0).cuda() # [b, num_cls+1]

        _, _, h, w = low_res_masks.shape
        masks = low_res_masks if (h,w) == self.mask_size else F.interpolate(low_res_masks, self.mask_size, mode="bilinear", align_corners=False)        
        
        outputs = {
            'preds': masks,
            'iou_predictions': iou_predictions
        }
        
        return outputs, prototypes, cls_embeddings, cls_ids


def cal_cls_embedding(sam_feats, masks, cls_ids, feat_size):
    sam_feats = rearrange(sam_feats, 'b (h w) c -> b c h w', h=feat_size, w=feat_size)
    sam_feats = F.interpolate(sam_feats.to(torch.float32), size=(feat_size*2, feat_size*2), mode='bilinear')
    masks = F.interpolate(masks.to(torch.float32), size=(feat_size*2, feat_size*2), mode='nearest')
    class_embeddings, i = [], 0
    
    for mask, ids in zip(masks, cls_ids):
        feat_slices = slice(i, i+len(ids))
        feats = sam_feats[feat_slices]
        
        for feat, cls_id in zip(feats, ids):
            mask_copy = torch.zeros_like(mask)
            mask_copy[mask == cls_id] = 255
        
            feat = feat.permute(1,2,0)
            class_embedding = feat[mask_copy.squeeze() > 0]
            class_embedding = class_embedding.mean(0).squeeze()
            class_embeddings.append(torch.nan_to_num(class_embedding))
    
    return torch.stack(class_embeddings).to('cuda')
   

class Prototype_Prompt_Encoder(nn.Module):
    def __init__(
            self, 
            feat_dim=256, 
            hidden_dim_dense=128, 
            hidden_dim_sparse=128, 
            feat_size=64, 
            num_tokens=8,
            num_classes=4
        ):      
        super(Prototype_Prompt_Encoder, self).__init__()
        self.dense_fc_1 = nn.Conv2d(feat_dim, hidden_dim_dense, 1)
        self.dense_fc_2 = nn.Conv2d(hidden_dim_dense, feat_dim, 1)
        
        self.relu = nn.ReLU()

        self.sparse_fc_1 = nn.Conv1d(feat_size*feat_size, hidden_dim_sparse, 1)
        self.sparse_fc_2 = nn.Conv1d(hidden_dim_sparse, num_tokens, 1)
                
        pn_cls_embeddings = [nn.Embedding(num_tokens, feat_dim) for _ in range(2)] # one for positive and one for negative 

        self.pn_cls_embeddings = nn.ModuleList(pn_cls_embeddings)
                
        self.feat_size = feat_size
        self.num_classes = num_classes        
            
    def forward(self, feat, prototypes, cls_ids):
  
        cls_prompts = prototypes.unsqueeze(-1) # [num_cls, 256, 1]
        cls_prompts = torch.stack([cls_prompts for _ in range(feat.size(0))], dim=0) # [b, num_cls, 256, 1]

        feat = torch.stack([feat for _ in range(cls_prompts.size(1))], dim=1) # [b, num_cls, 256, 256]

        # compute similarity matrix 
        sim = torch.matmul(feat, cls_prompts) # [b, num_cls, 256, 1]
        
        # compute class-activated feature
        feat = feat + feat*sim
        feat_sparse = feat.clone()
        
        # compute dense embeddings
        one_hot = torch.nn.functional.one_hot(cls_ids-1, self.num_classes) # cls_ids - 1 = 1; others = 0
        feat = feat[one_hot == 1] # [b, 256, 256]
        feat = rearrange(feat,'b (h w) c -> b c h w', h=self.feat_size, w=self.feat_size) # [b, 256, 16, 16]
        dense_embeddings = self.dense_fc_2(self.relu(self.dense_fc_1(feat))) # [b, 256, 16, 16]
        
        # compute sparse embeddings
        feat_sparse = rearrange(feat_sparse,'b num_cls hw c -> (b num_cls) hw c') # [b*num_cls, 256, 256]
        sparse_embeddings = self.sparse_fc_2(self.relu(self.sparse_fc_1(feat_sparse))) # [b*num_cls, num_cls, 256]
        sparse_embeddings = rearrange(sparse_embeddings,'(b num_cls) n c -> b num_cls n c', num_cls=self.num_classes) # [b, num_cls, num_tokens, 256]
        
        pos_embed = self.pn_cls_embeddings[1].weight.unsqueeze(0).unsqueeze(0) * one_hot.unsqueeze(-1).unsqueeze(-1)
        neg_embed = self.pn_cls_embeddings[0].weight.unsqueeze(0).unsqueeze(0) * (1-one_hot).unsqueeze(-1).unsqueeze(-1)

        sparse_embeddings = sparse_embeddings + pos_embed.detach() + neg_embed.detach()
        sparse_embeddings = rearrange(sparse_embeddings,'b num_cls n c -> b (num_cls n) c')
        
        return feat, dense_embeddings, sparse_embeddings
    

class Learnable_Prototypes(nn.Module):
    def __init__(self, num_classes=4 , feat_dim=256):
        super(Learnable_Prototypes, self).__init__()
        self.class_embeddings = nn.Embedding(num_classes, feat_dim)
        
    def forward(self):
        return self.class_embeddings.weight


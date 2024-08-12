# sabs
# lora
# 0.1
# CUDA_VISIBLE_DEVICES=3 python train.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.1 --sam_mode vit_b --sam_ckpt ../../data/experiments/weights/sam_vit_b_01ec64.pth --model_type lora --batch_size 16 --num_workers 16 --lr 5e-4

# w/o encoder
# 0.1
CUDA_VISIBLE_DEVICES=3 python train.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.1 --sam_mode vit_b --sam_ckpt ../../data/experiments/weights/sam_vit_b_01ec64.pth --model_type wo_encoder --batch_size 16 --num_workers 16 --lr 1e-3

# # 0.05
# CUDA_VISIBLE_DEVICES=3 python train.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.05 --sam_mode vit_b --sam_ckpt ../../data/experiments/weights/sam_vit_b_01ec64.pth --model_type wo_encoder --batch_size 16 --num_workers 16 --lr 5e-4

# # 0.01
# CUDA_VISIBLE_DEVICES=3 python train.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.01 --sam_mode vit_b --sam_ckpt ../../data/experiments/weights/sam_vit_b_01ec64.pth --model_type wo_encoder --batch_size 16 --num_workers 16 --lr 1e-4

######################

# bhx
# lora
# 0.1
# CUDA_VISIBLE_DEVICES=3 python train.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.1 --sam_mode vit_b --sam_ckpt ../../data/experiments/weights/sam_vit_b_01ec64.pth --model_type lora --batch_size 16 --num_workers 16 --lr 5e-4

# w/o encoder
# 0.1
CUDA_VISIBLE_DEVICES=3 python train.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.1 --sam_mode vit_b --sam_ckpt ../../data/experiments/weights/sam_vit_b_01ec64.pth --model_type wo_encoder --batch_size 16 --num_workers 16 --lr 1e-3

# # 0.05
# CUDA_VISIBLE_DEVICES=3 python train.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.05 --sam_mode vit_b --sam_ckpt ../../data/experiments/weights/sam_vit_b_01ec64.pth --model_type wo_encoder --batch_size 16 --num_workers 16 --lr 4e-4

# # 0.01
# CUDA_VISIBLE_DEVICES=3 python train.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.01 --sam_mode vit_b --sam_ckpt ../../data/experiments/weights/sam_vit_b_01ec64.pth --model_type wo_encoder --batch_size 16 --num_workers 16 --lr 1e-4

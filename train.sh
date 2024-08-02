# bhx
# 0.1
CUDA_VISIBLE_DEVICES=0 python train.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.1 --sam_mode vit_b --sam_ckpt ../../data/experiments/weights/sam_vit_b_01ec64.pth --model_type lora --batch_size 16 --num_workers 16 --lr 0.002

CUDA_VISIBLE_DEVICES=0 python train.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.1 --sam_mode vit_b --sam_ckpt ../../data/experiments/weights/sam_vit_b_01ec64.pth --model_type lora --batch_size 16 --num_workers 16 --lr 0.001

######################

# sabs
# 0.1
CUDA_VISIBLE_DEVICES=1 python train.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.1 --sam_mode vit_b --sam_ckpt ../../data/experiments/weights/sam_vit_b_01ec64.pth --model_type lora --batch_size 12 --num_workers 16 --lr 0.002

CUDA_VISIBLE_DEVICES=1 python train.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.1 --sam_mode vit_b --sam_ckpt ../../data/experiments/weights/sam_vit_b_01ec64.pth --model_type lora --batch_size 12 --num_workers 16 --lr 0.001

# bhx
# lora
# 0.1
# CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time  --vis True
# CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time  --volume True

# w/o encoder
# 0.1
CUDA_VISIBLE_DEVICES=3 python infer.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.1 --sam_mode vit_b --model_type wo_encoder --train_time 20240810-0001 --vis True
CUDA_VISIBLE_DEVICES=3 python infer.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.1 --sam_mode vit_b --model_type wo_encoder --train_time 20240810-0001 --volume True
CUDA_VISIBLE_DEVICES=3 python infer.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.1 --sam_mode vit_b --model_type wo_encoder --train_time 20240810-0001 --tsne True

# # 0.05
# CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.05 --sam_mode vit_b --model_type wo_encoder --train_time  --vis True
# CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.05 --sam_mode vit_b --model_type wo_encoder --train_time  --volume True
# CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.05 --sam_mode vit_b --model_type wo_encoder --train_time  --tsne True

# # 0.01
# CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.01 --sam_mode vit_b --model_type wo_encoder --train_time  --vis True
# CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.01 --sam_mode vit_b --model_type wo_encoder --train_time  --volume True
# CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.01 --sam_mode vit_b --model_type wo_encoder --train_time  --tsne True

###################

# sabs
# lora
# 0.1
# CUDA_VISIBLE_DEVICES=1 python infer.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time  --vis True
# CUDA_VISIBLE_DEVICES=1 python infer.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time  --volume True

# # w/o encoder
# # 0.1
# CUDA_VISIBLE_DEVICES=1 python infer.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.1 --sam_mode vit_b --model_type wo_encoder --train_time 20240809-1632 --vis True
# CUDA_VISIBLE_DEVICES=1 python infer.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.1 --sam_mode vit_b --model_type wo_encoder --train_time 20240809-1632 --volume True
# CUDA_VISIBLE_DEVICES=1 python infer.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.1 --sam_mode vit_b --model_type wo_encoder --train_time 20240809-1632 --tsne True

# # 0.05
# CUDA_VISIBLE_DEVICES=1 python infer.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.05 --sam_mode vit_b --model_type wo_encoder --train_time  --vis True
# CUDA_VISIBLE_DEVICES=1 python infer.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.05 --sam_mode vit_b --model_type wo_encoder --train_time  --volume True
# CUDA_VISIBLE_DEVICES=1 python infer.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.05 --sam_mode vit_b --model_type wo_encoder --train_time  --tsne True

# # 0.01
# CUDA_VISIBLE_DEVICES=1 python infer.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.01 --sam_mode vit_b --model_type wo_encoder --train_time  --vis True
# CUDA_VISIBLE_DEVICES=1 python infer.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.01 --sam_mode vit_b --model_type wo_encoder --train_time  --volume True
# CUDA_VISIBLE_DEVICES=1 python infer.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.01 --sam_mode vit_b --model_type wo_encoder --train_time  --tsne True

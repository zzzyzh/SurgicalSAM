# bhx
# lora
# 0.1
# CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 5 --task ven --dataset bhx_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time  --vis True
# CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 5 --task ven --dataset bhx_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time  --volume True

# w/o encoder
# 0.1
CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 5 --task ven --dataset bhx_sammed --scale 0.1 --sam_mode vit_b --model_type wo_encoder --train_time  --vis True
CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 5 --task ven --dataset bhx_sammed --scale 0.1 --sam_mode vit_b --model_type wo_encoder --train_time  --volume True
CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 5 --task ven --dataset bhx_sammed --scale 0.1 --sam_mode vit_b --model_type wo_encoder --train_time  --tsne True

# 0.05
CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 5 --task ven --dataset bhx_sammed --scale 0.05 --sam_mode vit_b --model_type wo_encoder --train_time  --vis True
CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 5 --task ven --dataset bhx_sammed --scale 0.05 --sam_mode vit_b --model_type wo_encoder --train_time  --volume True
CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 5 --task ven --dataset bhx_sammed --scale 0.05 --sam_mode vit_b --model_type wo_encoder --train_time  --tsne True

# 0.01
CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 5 --task ven --dataset bhx_sammed --scale 0.01 --sam_mode vit_b --model_type wo_encoder --train_time  --vis True
CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 5 --task ven --dataset bhx_sammed --scale 0.01 --sam_mode vit_b --model_type wo_encoder --train_time  --volume True
CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 5 --task ven --dataset bhx_sammed --scale 0.01 --sam_mode vit_b --model_type wo_encoder --train_time  --tsne True

###################

# sabs
# lora
# 0.1
# CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 9 --task abdomen --dataset sabs_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time  --vis True
# CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 9 --task abdomen --dataset sabs_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time  --volume True

# w/o encoder
# 0.1
CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 9 --task abdomen --dataset sabs_sammed --scale 0.1 --sam_mode vit_b --model_type wo_encoder --train_time  --vis True
CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 9 --task abdomen --dataset sabs_sammed --scale 0.1 --sam_mode vit_b --model_type wo_encoder --train_time  --volume True
CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 9 --task abdomen --dataset sabs_sammed --scale 0.1 --sam_mode vit_b --model_type wo_encoder --train_time  --tsne True

# 0.05
CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 9 --task abdomen --dataset sabs_sammed --scale 0.05 --sam_mode vit_b --model_type wo_encoder --train_time  --vis True
CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 9 --task abdomen --dataset sabs_sammed --scale 0.05 --sam_mode vit_b --model_type wo_encoder --train_time  --volume True
CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 9 --task abdomen --dataset sabs_sammed --scale 0.05 --sam_mode vit_b --model_type wo_encoder --train_time  --tsne True

# 0.01
CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 9 --task abdomen --dataset sabs_sammed --scale 0.01 --sam_mode vit_b --model_type wo_encoder --train_time  --vis True
CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 9 --task abdomen --dataset sabs_sammed --scale 0.01 --sam_mode vit_b --model_type wo_encoder --train_time  --volume True
CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 9 --task abdomen --dataset sabs_sammed --scale 0.01 --sam_mode vit_b --model_type wo_encoder --train_time  --tsne True

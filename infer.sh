# bhx
# 0.1
CUDA_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time  --vis True
CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time  --volume True

CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time  --vis True
CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time  --volume True

###################

# sabs
# 0.1
CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time  --vis True
CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time  --volume True

CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time  --vis True
CUDA_VISIBLE_DEVICES=0 python infer.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time  --volume True

# bhx
# 0.1
python infer.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time 20240710-1450 --vis True
python infer.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time 20240710-1450 --vis True --volume True

# 0.05
python infer.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.05 --sam_mode vit_b --model_type lora --train_time 20240711-1737 --vis True
python infer.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.05 --sam_mode vit_b --model_type lora --train_time 20240711-1737 --vis True --volume True

# 0.01
python infer.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.01 --sam_mode vit_b --model_type lora --train_time 20240712-1805 --vis True
python infer.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.01 --sam_mode vit_b --model_type lora --train_time 20240712-1805 --vis True --volume True

###################

# sabs
# 0.1
python infer.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time 20240711-1210 --vis True
python infer.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time 20240711-1210 --vis True --volume True

# 0.05
python infer.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.05 --sam_mode vit_b --model_type lora --train_time 20240713-2346 --vis True
python infer.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.05 --sam_mode vit_b --model_type lora --train_time 20240713-2346 --vis True --volume True

# 0.01
python infer.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.01 --sam_mode vit_b --model_type lora --train_time 20240714-0644 --vis True
python infer.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.01 --sam_mode vit_b --model_type lora --train_time 20240714-0644 --vis True --volume True

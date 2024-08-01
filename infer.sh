# bhx
# 0.1
python infer.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time  --vis True
python infer.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time  --volume True

python infer.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time  --vis True
python infer.py --resolution 512 --num_classes 4 --task ven --dataset bhx_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time  --volume True

###################

# sabs
# 0.1
python infer.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time  --vis True
python infer.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time  --volume True

python infer.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time  --vis True
python infer.py --resolution 512 --num_classes 8 --task abdomen --dataset sabs_sammed --scale 0.1 --sam_mode vit_b --model_type lora --train_time  --volume True

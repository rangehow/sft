CUDA_VISIBLE_DEVICES=1 python /data/ruanjh/best_training_method/sft/special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --alpha 0.8

CUDA_VISIBLE_DEVICES=2 python /data/ruanjh/best_training_method/sft/special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --alpha 1

CUDA_VISIBLE_DEVICES=3 python /data/ruanjh/best_training_method/sft/special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --alpha 0.5
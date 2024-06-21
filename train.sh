CUDA_VISIBLE_DEVICES=1 python special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --alpha 0.8 &

CUDA_VISIBLE_DEVICES=2 python special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --alpha 1 &

CUDA_VISIBLE_DEVICES=3 python special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --alpha 0.5 --div_mode False &


CUDA_VISIBLE_DEVICES=3 python special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --alpha 0 &

CUDA_VISIBLE_DEVICES=1 python special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --alpha 0.1 &
CUDA_VISIBLE_DEVICES=2 python special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --alpha 0.2 &
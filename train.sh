CUDA_VISIBLE_DEVICES=1 python special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --alpha 0.8 &

CUDA_VISIBLE_DEVICES=2 python special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --alpha 1 &

CUDA_VISIBLE_DEVICES=3 python special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --alpha 0.5 --div_mode False &


CUDA_VISIBLE_DEVICES=3 python special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --alpha 0 &

CUDA_VISIBLE_DEVICES=1 python special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --alpha 0.1 &
CUDA_VISIBLE_DEVICES=2 python special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --alpha 0.2 &




CUDA_VISIBLE_DEVICES=1 python special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --alpha 0.5 --div_mode False --weighted True&
CUDA_VISIBLE_DEVICES=2 python special_train.py --gradient_accumulation_steps 64 --total_bsz 256 --zero_prob 0 --alpha 0.7 --div_mode False --weighted True --dataset alpaca_gpt4,code,math&
CUDA_VISIBLE_DEVICES=3 python special_train.py --gradient_accumulation_steps 64 --total_bsz 256 --zero_prob 0 --alpha 0.3 --div_mode False --weighted True --dataset alpaca_gpt4,code,math&
# CUDA_VISIBLE_DEVICES=3 python special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --alpha 0.8 --div_mode False &
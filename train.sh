CUDA_VISIBLE_DEVICES=1 python special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --alpha 0.8 &

CUDA_VISIBLE_DEVICES=2 python special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --alpha 1 &

CUDA_VISIBLE_DEVICES=3 python special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --alpha 0.5 --div_mode False &


CUDA_VISIBLE_DEVICES=3 python special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --alpha 0 &

CUDA_VISIBLE_DEVICES=1 python special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --alpha 0.1 &
CUDA_VISIBLE_DEVICES=2 python special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --alpha 0.2 &




CUDA_VISIBLE_DEVICES=1 python special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --alpha 0.5 --div_mode False --weighted True&


CUDA_VISIBLE_DEVICES=2 python special_train.py --gradient_accumulation_steps 512 --total_bsz 512 --zero_prob 0 --alpha 0.5 --div_mode True --weighted True --dataset alpaca_gpt4,code,math&
CUDA_VISIBLE_DEVICES=3 python special_train.py --gradient_accumulation_steps 512 --total_bsz 512 --zero_prob 0 --alpha 0.5 --div_mode True --weighted False --dataset alpaca_gpt4,code,math --mix True --mix_ratio 0.5&

CUDA_VISIBLE_DEVICES=1 python special_train.py --gradient_accumulation_steps 512 --total_bsz 512 --zero_prob 0 --div_mode True --weighted True --dataset alpaca_gpt4,code,math --mix True --mix_ratio 0.5&
# CUDA_VISIBLE_DEVICES=3 python special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --alpha 0.8 --div_mode False &

CUDA_VISIBLE_DEVICES=1 python special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --div_mode True --weighted True --dataset alpaca_gpt4 --mix True --mix_ratio 0.8&
CUDA_VISIBLE_DEVICES=2 python special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --div_mode False --weighted True --dataset alpaca_gpt4 --mix True --mix_ratio 0.8&
CUDA_VISIBLE_DEVICES=3 python special_train.py --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --div_mode False --dataset alpaca_gpt4 --mix True --mix_ratio 0.8&
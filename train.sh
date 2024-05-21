
CUDA_VISIBLE_DEVICES=2 python special_train.py --model gemma_2b --dataset alpaca_gpt4 --output_dir ../output_test --div_mode True &



CUDA_VISIBLE_DEVICES=2 python special_train.py --model gemma_2b --dataset alpaca_gpt4 --output_dir ../stable_new_521  &


CUDA_VISIBLE_DEVICES=1 python special_train.py --model gemma_2b --dataset alpaca_gpt4 --output_dir ../stable_old_521

CUDA_VISIBLE_DEVICES=3 python special_train.py --model gemma_2b --dataset alpaca_gpt4 --output_dir ../zero_prob01woweight
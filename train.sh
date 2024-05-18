
CUDA_VISIBLE_DEVICES=2 python special_train.py --model gemma_2b --dataset alpaca_gpt4 --output_dir ../output_test --div_mode True &



CUDA_VISIBLE_DEVICES=1 python special_train.py --model gemma_2b --dataset alpaca_gpt4 --output_dir ../output_test --div_mode True --weighted &
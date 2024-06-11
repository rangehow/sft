
CUDA_VISIBLE_DEVICES=2 python special_train.py --model gemma_2b --dataset alpaca_gpt4  --div_mode True &



CUDA_VISIBLE_DEVICES=1 python special_train.py --model gemma_2b --dataset alpaca_gpt4  --div_mode False --zero_prob 0
CUDA_VISIBLE_DEVICES=2 python special_train.py --model gemma_2b --dataset alpaca_gpt4  --div_mode False --zero_prob 0.1
CUDA_VISIBLE_DEVICES=3 python special_train.py --model gemma_2b --dataset alpaca_gpt4  --div_mode False --zero_prob 0.1 --weighted



CUDA_VISIBLE_DEVICES=2  python  special_train.py --model gemma_2b --dataset alpaca_gpt4  --weighted --zero_prob 0 

CUDA_VISIBLE_DEVICES=1  python  special_train.py --model llama3_8b --dataset alpaca_gpt4  --weighted --zero_prob 0  --lora

CUDA_VISIBLE_DEVICES=2  python  special_train.py --model llama3_8b --dataset alpaca_gpt4  --weighted --zero_prob 0.1  --lora

CUDA_VISIBLE_DEVICES=2 python special_train.py --model gemma_2b --dataset alpaca_gpt4 --output_dir ../output_test --div_mode True &













CUDA_VISIBLE_DEVICES=1 python special_train.py --model gemma_2b --dataset alpaca_gpt4 --output_dir --weighted ../model_output/521weighted &


CUDA_VISIBLE_DEVICES=2 python special_train.py --model gemma_2b --dataset alpaca_gpt4 --output_dir  ../model_output/521zeroprob0  --zero_prob 0 &




CUDA_VISIBLE_DEVICES=3 python special_train.py --model gemma_2b --dataset alpaca_gpt4 --output_dir ../model_output/521origin&
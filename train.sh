
CUDA_VISIBLE_DEVICES=2 python special_train.py --model gemma_2b --dataset alpaca_gpt4 --output_dir ../output_test --div_mode True &









CUDA_VISIBLE_DEVICES=2  python  special_train.py --model gemma_2b --dataset alpaca_gpt4  --weighted --zero_prob 0 

CUDA_VISIBLE_DEVICES=1  python  special_train.py --model gemma_2b --dataset alpaca_gpt4  --weighted --zero_prob 0 

CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --config_file 8b3card.yaml  special_train.py --model gemma_2b --dataset alpaca_gpt4  --weighted --zero_prob 0 


CUDA_VISIBLE_DEVICES=2 python special_train.py --model gemma_2b --dataset alpaca_gpt4 --output_dir  ../model_output/523zeroprob02  --zero_prob 0.2 &




CUDA_VISIBLE_DEVICES=3 python special_train.py --model gemma_2b --dataset alpaca_gpt4 --output_dir ../model_output/523origin&
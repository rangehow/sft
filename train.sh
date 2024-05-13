export CUDA_VISIBLE_DEVICES=1,2,3
torchrun --nproc-per-node 3 special_train.py --model gemma_2b --dataset alpaca_gpt4 --output_dir ../output
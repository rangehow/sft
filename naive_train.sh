# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 /data/ruanjh/best_training_method/sft/naive_train.py \
#     --model gemma_2b \
#     --dataset alpaca_cleaned \
#     --output_dir /data/ruanjh/best_training_method/output \

CUDA_VISIBLE_DEVICES=0,1,2,3 python /data/ruanjh/best_training_method/sft/naive_train.py \
    --model gemma_2b \
    --dataset alpaca_cleaned \
    --output_dir /data/ruanjh/best_training_method/output \
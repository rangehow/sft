# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 /data/ruanjh/best_training_method/sft/naive_train.py \
#     --model gemma_2b \
#     --dataset alpaca_cleaned \
#     --output_dir /data/ruanjh/best_training_method/output \

CUDA_VISIBLE_DEVICES=2 python  naive_train.py \
    --model gemma_2b \
    --dataset alpaca_gpt4 \
    --train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --output_dir gemma_naive_6m2d \
    # --lora \
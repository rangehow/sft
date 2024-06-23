# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 /data/ruanjh/best_training_method/sft/naive_train.py \
#     --model gemma_2b \
#     --dataset alpaca_cleaned \
#     --output_dir /data/ruanjh/best_training_method/output \

CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc-per-node 2  naive_train.py \
    --model gemma_2b \
    --dataset alpaca_gpt4,code,math \
    --total_bsz 256 \
    --gradient_accumulation_steps 64 \
    --output_dir gemma_naive_bsz256_mix \
    # --label_smoothing_factor 0.1 \



CUDA_VISIBLE_DEVICES=1 python naive_train.py \
    --model gemma_2b \
    --dataset alpaca_gpt4,code,math \
    --total_bsz 256 \
    --gradient_accumulation_steps 64 \
    --output_dir gemma_naive_bsz256_mix \
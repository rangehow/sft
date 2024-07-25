# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 /data/ruanjh/best_training_method/sft/naive_train.py \
#     --model gemma_2b \
#     --dataset alpaca_cleaned \
#     --output_dir /data/ruanjh/best_training_method/output \

torchrun --nproc-per-node 4 naive_train.py \
    --model gemma_2b \
    --dataset alpaca_gpt4,code,math \
    --total_bsz 512 \
    --gradient_accumulation_steps 64 \
    --num_train_epochs 3\
    --label_smoothing_factor 0.1 \
    # --lora


# CUDA_VISIBLE_DEVICES=2,3 torchrun naive_train.py \
#     --model llama3_8b \
#     --dataset alpaca_gpt4,code,math \
#     --total_bsz 512 \
#     --gradient_accumulation_steps 256 \
#     --num_train_epochs 3\
#     --w_template False \
#     --lora


# CUDA_VISIBLE_DEVICES=0,1 torchrun  --nproc-per-node 2 --master-port 29873  naive_train.py \
#     --model llama3_8b \
#     --dataset alpaca_gpt4,code,math \
#     --total_bsz 512 \
#     --gradient_accumulation_steps 256 \
#     --num_train_epochs 3\
#     --w_template True \
#     --lora


# CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc-per-node 2 naive_train.py \
#     --model Qwen/Qwen2-1.5B \
#     --dataset alpaca_gpt4,code,math \
#     --total_bsz 512 \
#     --gradient_accumulation_steps 256 \
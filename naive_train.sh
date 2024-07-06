# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc-per-node 4 /data/ruanjh/best_training_method/sft/naive_train.py \
#     --model gemma_2b \
#     --dataset alpaca_cleaned \
#     --output_dir /data/ruanjh/best_training_method/output \

CUDA_VISIBLE_DEVICES=1 python naive_train.py \
    --model llama3_8b \
    --dataset alpaca_gpt4,code,math \
    --total_bsz 512 \
    --gradient_accumulation_steps 512 \
    --output_dir llama_naive_bsz512_mix_ls01 \
    --label_smoothing_factor 0.1 \
    --lora



CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc-per-node 2 naive_train.py \
    --model Qwen/Qwen2-1.5B \
    --dataset alpaca_gpt4,code,math \
    --total_bsz 512 \
    --gradient_accumulation_steps 256 \
    --output_dir qwen2_1.5B_naive_bsz512_mix_2card_acc256 \
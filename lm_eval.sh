#!/bin/bash


models=(
    '/niutrans/NEUNLP/rjh/sft/gemma_2b_alpaca_gpt4_5m30d_0_weighted_div'
    '/niutrans/NEUNLP/rjh/sft/gemma_naive_6m2d'
    '/niutrans/NEUNLP/rjh/models/gemma-1.1-2b-it'
    '/niutrans/NEUNLP/rjh/models/gemma-2b'
    '/niutrans/NEUNLP/rjh/models/gemma-2b-it'
)

# 遍历每个模型并执行命令
for model in "${models[@]}"; do
    accelerate launch --config_file lm_eval.yaml -m lm_eval --model hf \
        --model_args pretrained="$model" \
        --tasks triviaqa \
        --batch_size auto
done

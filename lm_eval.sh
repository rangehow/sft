#!/bin/bash

# 定义模型列表
models=(
    '/niutrans/NEUNLP/rjh/sft/gemma_2b_alpaca_gpt4_5m30d_0_weighted_div'
    '/niutrans/NEUNLP/rjh/sft/gemma_naive_6m2d'
    '/niutrans/NEUNLP/rjh/models/gemma-1.1-2b-it'
    '/niutrans/NEUNLP/rjh/models/gemma-2b'
    '/niutrans/NEUNLP/rjh/models/gemma-2b-it'
)

# 定义任务列表和对应的 num_fewshot
tasks=(
    "triviaqa 5"
    "gsm8k_cot 0"
    "agieval 0"
    "mmlu 0"
    "truthfulqa_mc2 0"
    "bbh_cot_fewshot 0"
    "arc_challenge 0"
    "drop 0"
)

# 遍历每个模型和任务并执行命令
for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        # 解析任务和 num_fewshot
        IFS=' ' read -r task_name num_fewshot <<< "$task"
        
        # 执行命令
        accelerate launch --config_file lm_eval.yaml -m lm_eval --model hf \
            --model_args pretrained="$model" \
            --tasks "$task_name" \
            --batch_size 8 \
            --num_fewshot "$num_fewshot"
    done
done

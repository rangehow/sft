#!/bin/bash

{
# 定义模型列表
models=(
    # '/niutrans/NEUNLP/rjh/sft/gemma_2b_alpaca_gpt4_5m30d_0_weighted_div'
    # '/niutrans/NEUNLP/rjh/sft/gemma_naive_6m2d'
    # '/niutrans/NEUNLP/rjh/models/gemma-1.1-2b-it'
    # '/niutrans/NEUNLP/rjh/models/gemma-2b'
    # '/niutrans/NEUNLP/rjh/models/gemma-2b-it'
    '/niutrans/NEUNLP/rjh/sft/gemma_naive_6m9d_ls0.1'
)

# models=(
#     '/niutrans/NEUNLP/rjh/sft/llama3_8b_alpaca_gpt4_6m2d_0_bsz64_weighted_div_lora' # r=32
#     '/niutrans/NEUNLP/rjh/sft/llama3_8b_alpaca_gpt4_6m2d_0_bsz64_weighted_div_lora_save' # r=8
#     '/niutrans/NEUNLP/rjh/models/Llama-3-8B'
#     '/niutrans/NEUNLP/rjh/models/Meta-Llama-3-8B-Instruct'
# )




timestamp=$(date +"%Y%m%d_%H%M%S")

#     "drop 0"
#    "mmlu 0"
# 定义任务列表和对应的 num_fewshot
tasks=(
    "mmlu 0"
    "gsm8k 0"
    "humaneval 0"
    "triviaqa 5"
    "agieval 0"
    "truthfulqa_mc2 0"
    "bbh_cot_fewshot 3"
    "arc_challenge 0"
    "winogrande 5"
    "sciq 0"
)

# 遍历每个模型和任务并执行命令
for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        # 解析任务和 num_fewshot
        IFS=' ' read -r task_name num_fewshot <<< "$task"
        echo "$task_name"
        if [ "$task_name" == "mmlu" ] || [ "$task_name" == "gsm8k" ] || [ "$task_name" == "humaneval" ]; then
            CUDA_VISIBLE_DEVICES=1,2,3  python -m sft.eval.gsm8k  --reuse --mode 0 --shot --dp --dataset "$task_name" --model "$model" --output_path  "$(dirname "$(realpath "$0")")/${timestamp}/"
        else
            # 执行命令
            accelerate launch --config_file sft/lm_eval.yaml -m lm_eval --model hf \
                --model_args pretrained="$model" \
                --tasks "$task_name" \
                --batch_size 8 \
                --num_fewshot "$num_fewshot" \
                --output_path  "$(dirname "$(realpath "$0")")/${timestamp}/"


        fi
        
    done
done

# 代码测试应该如何进行？
# Human-Eval

# accelerate launch --config_file lm_eval.yaml -m lm_eval --model hf \
#             --model_args pretrained='/niutrans/NEUNLP/rjh/models/gemma-2b-it' \
#             --tasks winogrande \
#             --batch_size 8 \
#             --num_fewshot 5

} >> output.txt
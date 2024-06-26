#!/bin/bash


# 定义模型列表
models=(
    # '/niutrans/NEUNLP/rjh/sft/gemma_2b_alpaca_gpt4_5m30d_0_weighted_div'
    # '/niutrans/NEUNLP/rjh/sft/gemma_naive_6m2d'
    # '/niutrans/NEUNLP/rjh/models/gemma-1.1-2b-it'
    # '/niutrans/NEUNLP/rjh/models/gemma-2b'
    # '/niutrans/NEUNLP/rjh/models/gemma-2b-it'
    # '/niutrans/NEUNLP/rjh/sft/gemma_naive_6m9d_ls0.1'
    # 'sft/gemma_naive_bsz256'
    # 'sft/gemma_2b_alpaca_gpt4_6m14d_0_bsz256_alpha0.5_div'                          
    # 'sft/gemma_2b_alpaca_gpt4_6m14d_0_bsz256_alpha0.8_div'                          
    # 'sft/gemma_2b_alpaca_gpt4_6m14d_0_bsz256_alpha1_div'
    # 'sft/gemma_2b_alpaca_gpt4_6m16d_0_bsz256_alpha0.1_div/'
    # 'sft/gemma_2b_alpaca_gpt4_6m16d_0_bsz256_alpha0.2_div/'
    # 'sft/gemma_2b_alpaca_gpt4_6m16d_0_bsz256_alpha0_div/'
    # 'sft/gemma_2b_alpaca_gpt4_6m21d_0_bsz256_alpha0.5'
    # 'sft/gemma_2b_alpaca_gpt4_6m21d_0_bsz256_alpha1'
    # 'sft/gemma_2b_alpaca_gpt4_6m21d_0_bsz256_alpha0'
    # 'sft/gemma_2b_alpaca_gpt4_6m22d_0_bsz256_alpha1_weighted/'
    # 'sft/gemma_2b_alpaca_gpt4_6m22d_0_bsz256_alpha0.8_weighted/'
    # 'sft/gemma_naive_bsz512_mix'
    'sft/gemma_2b_alpaca_gpt4_6m26d_0_bsz256_alpha0.8_weighted_div_mix0.8/'
    'sft/gemma_2b_alpaca_gpt4_6m26d_0_bsz256_alpha0.8_weighted_mix0.8/'


)

# models=(
#     '/niutrans/NEUNLP/rjh/sft/llama3_8b_alpaca_gpt4_6m2d_0_bsz64_weighted_div_lora' # r=32
#     '/niutrans/NEUNLP/rjh/sft/llama3_8b_alpaca_gpt4_6m2d_0_bsz64_weighted_div_lora_save' # r=8
#     '/niutrans/NEUNLP/rjh/models/Llama-3-8B'
#     '/niutrans/NEUNLP/rjh/models/Meta-Llama-3-8B-Instruct'
# )
model_string=""
for model in "${models[@]}"; do
    # 如果result不为空，则添加逗号
    if [ -n "$model_string" ]; then
        model_string+=","
    fi
    # 添加模型路径到result
    model_string+="$model"
done

# 输出结果
echo "$model_string"



timestamp=$(date +"%Y%m%d_%H%M%S")

#     "drop 0"
#    "mmlu 0"
# 定义任务列表和对应的 num_fewshot
tasks=(
    "truthfulqa_mc2 0"
    "bbh_cot_fewshot 3"
    "arc_challenge 0"
    "triviaqa 5"
    "agieval 0"
    "sciq 0"
    "winogrande 5"
    "ifeval 0"
)
 CUDA_VISIBLE_DEVICES=1,2,3  python -m sft.eval.gsm8k  --reuse --mode 0 --shot --dp --dataset mmlu,gsm8k,humaneval --model "${model_string}" --output_path  "$(dirname "$(realpath "$0")")/${timestamp}/"
# 遍历每个模型和任务并执行命令
for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        # 解析任务和 num_fewshot
        IFS=' ' read -r task_name num_fewshot <<< "$task"
        echo "$task_name"
        
        # 执行命令
        accelerate launch --config_file sft/lm_eval.yaml -m lm_eval --model hf \
            --model_args pretrained="$model" \
            --tasks "$task_name" \
            --batch_size 12 \
            --num_fewshot "$num_fewshot" \
            --output_path  "$(dirname "$(realpath "$0")")/${timestamp}/${model}"
      
    done
done



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
    
    # 'sft/gemma_2b_alpaca_gpt4_6m26d_0_bsz256_alpha0.8_mix0.8'
    # 'sft/gemma_2b_alpaca_gpt4_6m27d_0_bsz256_alpha0.8_mix0.5'
    # 'sft/qwen2_1.5B_naive_bsz512_mix_2card_acc256'
    # 'sft/gemma_2b_alpaca_gpt4_6m27d_0_bsz256_alpha0.8_mix0.2'
    # 'sft/gemma_2b_alpaca_gpt4_6m27d_0_bsz256_alpha0.8_weighted_mix0.8'
    
    # 'sft/llama3_8b_alpaca_gpt4_math_code_7m2d_0_bsz512_alpha0.8_mix0.8_lora/checkpoint-1221'
    # 'sft/llama_naive_bsz512_mix/checkpoint-1293'
    # 'sft/llama3_8b_alpaca_gpt4_math_code_7m2d_0_bsz512_alpha0.8_mix0.8_lora/checkpoint-407'
    # 'sft/llama3_8b_alpaca_gpt4_math_code_7m2d_0_bsz512_alpha0.8_mix0.8_lora/checkpoint-814'
    # 'sft/llama_naive_bsz512_mix/checkpoint-863'
    # 'sft/llama_naive_bsz512_mix/checkpoint-431'
    # 'models/Llama-3-8B'
    # 'sft/llama3_8b_alpaca_gpt4_math_code_7m5d_0_bsz512_alpha0.8_mix0.5_lora/checkpoint-1221'
    # 'sft/llama3_8b_alpaca_gpt4_math_code_7m5d_0_bsz512_alpha0.8_mix0.5_lora/checkpoint-814'
    # 'sft/llama3_8b_alpaca_gpt4_math_code_7m5d_0_bsz512_alpha0.8_mix0.5_lora/checkpoint-407'
    # 'sft/llama_naive_bsz512_mix_ls01/checkpoint-1293'
    # 'sft/llama_naive_bsz512_mix_ls01/checkpoint-863'
    # 'sft/llama_naive_bsz512_mix_ls01/checkpoint-431'
    # 'sft/llama_naive_bsz512_mix_ls01/checkpoint-1295'
    # 'sft/llama_naive_bsz512_mix_ls01/checkpoint-1726'
    # 'sft/llama_naive_bsz512_mix_ls01/checkpoint-2158'
    # 'sft/llama_naive_bsz512_mix_ls01/checkpoint-2590'
    # 'sft/llama_naive_bsz512_mix_ls01/checkpoint-3021'
    # 'sft/llama_naive_bsz512_mix_ls01/checkpoint-3453'
    # 'sft/llama_naive_bsz512_mix_ls01/checkpoint-3879'
    # 'sft/llama3_8b_alpaca_gpt4_math_code_7m8d_0_bsz512_alpha0.8_mix0.8_lora/checkpoint-3663'
    # 'sft/llama3_8b_alpaca_gpt4_math_code_7m8d_0_bsz512_alpha0.8_mix0.8_lora/checkpoint-2443'
    # 'sft/llama3_8b_alpaca_gpt4_math_code_7m8d_0_bsz512_alpha0.8_mix0.8_lora/checkpoint-2850'
    # 'sft/llama3_8b_alpaca_gpt4_math_code_7m8d_0_bsz512_alpha0.8_mix0.8_lora/checkpoint-407'
    # 'sft/llama3_8b_alpaca_gpt4_math_code_7m8d_0_bsz512_alpha0.8_mix0.8_lora/checkpoint-814'
    # 'sft/llama3_8b_alpaca_gpt4_math_code_7m8d_0_bsz512_alpha0.8_mix0.8_lora/checkpoint-3258'
    # 'sft/llama3_8b_alpaca_gpt4_math_code_7m8d_0_bsz512_alpha0.8_mix0.8_lora/checkpoint-1221'
    # 'sft/llama3_8b_alpaca_gpt4_math_code_7m8d_0_bsz512_alpha0.8_mix0.8_lora/checkpoint-1629'
    # 'sft/llama3_8b_alpaca_gpt4_math_code_7m8d_0_bsz512_alpha0.8_mix0.8_lora/checkpoint-2036'

    sft/llama3_8b_alpaca_gpt4_math_code_7m18d_0_bsz512_alpha0.8_mix0.8_lora/checkpoint-407 
    sft/llama3_8b_alpaca_gpt4_math_code_7m18d_0_bsz512_alpha0.8_mix0.8_lora/checkpoint-814
    sft/llama3_8b_alpaca_gpt4_math_code_7m18d_0_bsz512_alpha0.8_mix0.8_lora/checkpoint-1221
    # sft/llama_naive_bsz512_mix/checkpoint-431
    # sft/llama_naive_bsz512_mix/checkpoint-863
    # sft/llama_naive_bsz512_mix/checkpoint-1293
)


# llama3_8b_alpaca_gpt4_math_code_7m16d_0.1_bsz256_alpha0.8_mix0.7_lora/checkpoint-407 
# llama3_8b_alpaca_gpt4_math_code_7m16d_0.1_bsz256_alpha0.8_mix0.7_lora/checkpoint-814
# llama3_8b_alpaca_gpt4_math_code_7m16d_0.1_bsz256_alpha0.8_mix0.7_lora/checkpoint-1221


    

    







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

python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset gsm8k,mmlu,humaneval --model "${model_string}"  --timestamp ${timestamp} --output_path  "$(dirname "$(realpath "$0")")/"





# gsm8k,mmlu,humaneval,


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
            --batch_size 'auto' \
            --num_fewshot "$num_fewshot" \
            --output_path  "$(dirname "$(realpath "$0")")/${timestamp}/${model}"
      
    done
done




# 定义任务列表和对应的 num_fewshot



# accelerate launch --config_file sft/lm_eval.yaml -m lm_eval --model hf \
#             --model_args pretrained="models/Llama-3-8B" \
#             --tasks "truthfulqa_mc2" \
#             --batch_size 'auto' \
#             --num_fewshot 0 \

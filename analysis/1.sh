#!/bin/bash

# 定义模型数组
models=("Phi-3-mini-4k-instruct" "Phi-3-small-8k-instruct" "Phi-3-medium-4k-instruct")
   
GPU_PAIRS=("0,1,2,3,4,5,6,7")
OUTPUT_PATH="/mnt/rangehow/rangehow/sft/analysis/result.json"
LOG_DIR="/mnt/rangehow/rangehow/sft/analysis/output"

# 遍历模型数组并运行命令
for model in "${models[@]}"; do
    CUDA_VISIBLE_DEVICES="${GPU_PAIRS[0]}" \
    python -m sft.analysis.kurtosis \
    --model "$model" \
    --output_path "$OUTPUT_PATH" \
    > "$LOG_DIR/${model}.log" 2>&1 &

    last_pid=$!

    # 等待该进程结束
    wait $last_pid
    if [ $? -eq 0 ]; then
        echo "$model 运行成功。"
    else
        echo "$model 运行失败，请检查错误。"
    fi
done
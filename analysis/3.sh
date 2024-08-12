#!/bin/bash

# 定义模型数组
models=("gemma-2-2b-it")
OUTPUT_PATH="/mnt/rangehow/rangehow/sft/analysis/result.json"
LOG_DIR="/mnt/rangehow/rangehow/sft/analysis/output"

# 遍历模型数组并运行命令
for model in "${models[@]}"; do
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python -m sft.analysis.kurtosis \
    --model "$model" \
    --output_path "$OUTPUT_PATH" \
    > "$LOG_DIR/${model}.log" 2>&1 &

    last_pid=$!

    # 等待该进程结束
    wait $last_pid
    # 检查每个命令是否成功执行
    if [ $? -eq 0 ]; then
        echo "$model 运行成功。"
    else
        echo "$model 运行失败，请检查错误。"
    fi
done
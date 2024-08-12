#!/bin/bash
declare -a MODELS=("Yi-1.5-34B-Chat")
DATASET="alpaca_gpt4,math,code"
OUTPUT_PATH="/mnt/rangehow/rangehow/sft/analysis/ngram0/result.json"
ZERO_PROB=0
DIV_MODE=False
LOG_DIR="/mnt/rangehow/rangehow/sft/analysis/ngram0"

declare -a GPU_PAIRS=("0,1,2,3")
declare -a MIX_RATIOS=("0.8")

declare -a TASK_PIDS=()

assign_task() {
    local MODEL=$1
    local GPU_PAIR=$2
    local MIX_RATIO=$3
    CUDA_VISIBLE_DEVICES="$GPU_PAIR" \
    python -m sft.analysis.logits \
    --model $MODEL \
    --dataset $DATASET \
    --output_path $OUTPUT_PATH \
    --zero_prob $ZERO_PROB \
    --div_mode $DIV_MODE \
    --mix_ratio $MIX_RATIO \
    --template yi \
    > $LOG_DIR/${MODEL}_dm_${DIV_MODE}-zp_${ZERO_PROB}-mr_${MIX_RATIO//./}.log 2>&1 &
    
    echo $!  # 返回任务的 PID
}

for MODEL in "${MODELS[@]}"; do
    for MIX_RATIO in "${MIX_RATIOS[@]}"; do
        while true; do
            for i in "${!GPU_PAIRS[@]}"; do
                if [ -z "${TASK_PIDS[$i]}" ] || ! kill -0 "${TASK_PIDS[$i]}" 2>/dev/null; then
                    echo "Assigning MODEL $MODEL with MIX_RATIO $MIX_RATIO to GPU_PAIR ${GPU_PAIRS[$i]}"
                    TASK_PIDS[$i]=$(assign_task "$MODEL" "${GPU_PAIRS[$i]}" "$MIX_RATIO")
                    break 2  # 跳出 while 和 for 循环
                fi
            done
            sleep 120  # 等待 2分钟再检查
        done
    done
done

for PID in "${TASK_PIDS[@]}"; do
    if [ -n "$PID" ]; then
        wait $PID
    fi
done

echo "All tasks completed."
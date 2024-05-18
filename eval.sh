  
export CUDA_VISIBLE_DEVICES=1 python -m sft.eval.gsm8k --dataset gsm8k --model ../models/gemma-2b/ --output ../eval_output/gsm8k_base_0shot.json --vllm --mode 0 &

export CUDA_VISIBLE_DEVICES=2 python -m sft.eval.gsm8k --dataset gsm8k --model ../output --output ../eval_output/gsm8k_my_0shot.json --vllm --mode 1&
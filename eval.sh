# 很麻烦，必须在sft上层文件夹才能运行
CUDA_VISIBLE_DEVICES=3 python -m sft.eval.gsm8k --dataset gsm8k --model models/gemma-2b/ --output eval_output/gsm8k_base_shot_test.json --vllm --mode 0 --shot&
CUDA_VISIBLE_DEVICES=2 python -m sft.eval.gsm8k --dataset gsm8k --model output --output eval_output/gsm8k_my_shot.json --vllm --mode 1 --shot &
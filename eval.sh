# 很麻烦，必须在sft上层文件夹才能运行
# CUDA_VISIBLE_DEVICES=1 python -m sft.eval.gsm8k --dataset gsm8k --model models/gemma-2b/ --output eval_output/gsm8k_base_shot.json --vllm --mode 0 --shot &
# CUDA_VISIBLE_DEVICES=2 python -m sft.eval.gsm8k --dataset gsm8k --model output --output eval_output/gsm8k_my_shot_wotemplate.json --vllm --mode 0 --shot &


# CUDA_VISIBLE_DEVICES=1 python -m sft.eval.gsm8k --dataset gsm8k --model /data/ruanjh/best_training_method/gemma-2b --vllm --mode 0 --shot 

# 521zeroprob0
# 521origin
# CUDA_VISIBLE_DEVICES=1 python -m sft.eval.gsm8k --dataset gsm8k --model model_output/521origin --vllm --mode 0 --shot 

# CUDA_VISIBLE_DEVICES=2,3 python -m sft.eval.gsm8k --dataset gsm8k --model /data/ruanjh/best_training_method/output/checkpoint-12500 --vllm --mode 0 --shot 
# CUDA_VISIBLE_DEVICES=0,1 python -m sft.eval.gsm8k --dataset gsm8k --model /data/ruanjh/best_training_method/gemma-2b  --vllm --mode 0 --shot 

CUDA_VISIBLE_DEVICES=2,3 python -m sft.eval.gsm8k --dataset gsm8k --model /data/ruanjh/best_training_method/output522/checkpoint-6470 --vllm --mode 1 --shot 



CUDA_VISIBLE_DEVICES=1 python -m sft.eval.gsm8k --model model_output/523zeroprob02 --dataset gsm8k --vllm --mode 0 --shot &


CUDA_VISIBLE_DEVICES=2 python -m sft.eval.gsm8k --model model_output/522zeroprob0 --dataset gsm8k --vllm --mode 0 --shot &




CUDA_VISIBLE_DEVICES=3 python -m sft.eval.gsm8k --model model_output/522origin --dataset gsm8k --vllm --mode 0 --shot&
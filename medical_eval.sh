CUDA_VISIBLE_DEVICES=0,1 python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medical --model /niutrans/NEUNLP/rjh/sft/llama_med_sft/checkpoint-32,/niutrans/NEUNLP/rjh/sft/llama_med_sft/checkpoint-64,/niutrans/NEUNLP/rjh/sft/llama_med_sft/checkpoint-96 --output_path  fuck






CUDA_VISIBLE_DEVICES=0,1 python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medical --model mistralai/Mistral-7B-v0.3 --output_path  fuck
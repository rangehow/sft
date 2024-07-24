CUDA_VISIBLE_DEVICES=0,1 python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medical --model /niutrans/NEUNLP/rjh/sft/llama_med_sft/checkpoint-32,/niutrans/NEUNLP/rjh/sft/llama_med_sft/checkpoint-64,/niutrans/NEUNLP/rjh/sft/llama_med_sft/checkpoint-96 --output_path  fuck






python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medical --model /niutrans/NEUNLP/rjh/models/Mistral-7B-v0.3 --output_path  fuck

python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medical --model /niutrans/NEUNLP/rjh/sft/naive_mistral_7b_textbooks_wiki_medical_medical_transcription_7m24d_bsz256_cosine_lr0.0006/checkpoint-535 --output_path  fuck


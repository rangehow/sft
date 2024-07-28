CUDA_VISIBLE_DEVICES=0,1 python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medical --model /niutrans/NEUNLP/rjh/sft/llama_med_sft/checkpoint-32,/niutrans/NEUNLP/rjh/sft/llama_med_sft/checkpoint-64,/niutrans/NEUNLP/rjh/sft/llama_med_sft/checkpoint-96 --output_path  fuck



python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medical --model /niutrans/NEUNLP/rjh/models/Llama-3-8B --output_path  fuck

python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medical --model /niutrans/NEUNLP/rjh/models/Qwen2-7B --output_path  fuck

python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medical --model /niutrans/NEUNLP/rjh/models/Meta-Llama-3-8B-Instruct --output_path  fuck


python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medical --model /niutrans/NEUNLP/rjh/models/Mistral-7B-v0.3 --output_path  fuck

python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medical --model /niutrans/NEUNLP/rjh/sft/naive_mistral_7b_textbooks_wiki_medical_medical_transcription_7m24d_bsz256_cosine_lr0.0006/checkpoint-535 --output_path  fuck

python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medical --model /niutrans/NEUNLP/rjh/sft/naive_llama3_8b_textbooks_wiki_medical_medical_transcription_7m27d_bsz256_cosine_lr1e-04_warmratio5e-02/checkpoint-535 --output_path  fuck


python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medical --model /niutrans/NEUNLP/rjh/sft/naive_llama3_8b_medquad_7m27d_bsz256_cosine_lr6e-04_template_warmratio5e-02 --output_path  fuck

python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medqa --model /niutrans/NEUNLP/rjh/sft/naive_qwen2_7b_textbooks_7m28d_bsz256_cosine_lr3e-04_warmratio5e-02 --output_path  fuck



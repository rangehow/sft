CUDA_VISIBLE_DEVICES=0,1 python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medical --model /niutrans/NEUNLP/rjh/sft/llama_med_sft/checkpoint-32,/niutrans/NEUNLP/rjh/sft/llama_med_sft/checkpoint-64,/niutrans/NEUNLP/rjh/sft/llama_med_sft/checkpoint-96 --output_path  fuck



python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medical --model /niutrans/NEUNLP/rjh/models/Llama-3-8B --output_path  fuck

python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medical --model /niutrans/NEUNLP/rjh/models/Qwen2-7B --output_path  fuck

python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medical --model /niutrans/NEUNLP/rjh/models/Meta-Llama-3-8B-Instruct --output_path  fuck


python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medical --model /niutrans/NEUNLP/rjh/models/Qwen2-7B --output_path  fuck

python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medical --model /niutrans/NEUNLP/rjh/sft/naive_mistral_7b_textbooks_wiki_medical_medical_transcription_7m24d_bsz256_cosine_lr0.0006/checkpoint-535 --output_path  fuck

python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medical --model /niutrans/NEUNLP/rjh/sft/naive_llama3_8b_textbooks_wiki_medical_medical_transcription_7m27d_bsz256_cosine_lr1e-04_warmratio5e-02/checkpoint-535 --output_path  fuck


python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medmcqa,pubmedqa,medqa --model /niutrans/NEUNLP/rjh/models/Qwen2-7B,/niutrans/NEUNLP/rjh/sft/naive_qwen2_7b_medquad_7m29d_bsz256_cosine_lr2e-05_template_warmratio5e-02,/niutrans/NEUNLP/rjh/sft/qwen2_7b_medquad_7m29d_0_bsz256_alpha0.8_cosine_lr2e-05_mix0.8_template_warmratio5e-02/checkpoint-63,/niutrans/NEUNLP/rjh/sft/qwen2_7b_medquad_7m29d_0_bsz256_alpha0.8_cosine_lr2e-05_mix0.8_template_warmratio5e-02/checkpoint-127,/niutrans/NEUNLP/rjh/sft/qwen2_7b_medquad_7m29d_0_bsz256_alpha0.8_cosine_lr2e-05_mix0.8_template_warmratio5e-02/checkpoint-189 --output_path  fuck

python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medical --model /niutrans/NEUNLP/rjh/sft/qwen2_7b_medquad_7m29d_0_bsz256_alpha0.8_cosine_lr2e-05_mix0.8_template_warmratio5e-02/checkpoint-127 --output_path  fuck

python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medmcqa,pubmedqa,medqa --model /niutrans/NEUNLP/rjh/models/Qwen2-7B --output_path  fuck

python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medqa,medmcqa,pubmedqa --model /niutrans/NEUNLP/rjh/sft/naive_qwen2_7b_medquad_7m29d_bsz256_cosine_lr2e-05_template_warmratio5e-02 --output_path  fuck

# ,
qwen2_7b_medquad_7m29d_0_bsz256_alpha0.8_cosine_lr2e-05_mix0.8_template_warmratio5e-02


python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medical --model /niutrans/NEUNLP/rjh/models/gemma-2-9b --output_path  fuck

python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medical --model /niutrans/NEUNLP/rjh/sft/naive_qwen2_7b_medquad_7m29d_bsz256_cosine_lr2e-05_template_warmratio5e-02,/niutrans/NEUNLP/rjh/sft/naive_qwen_med_pt_medquad_7m29d_bsz256_cosine_lr2e-05_template_warmratio5e-02 --output_path  fuck


# med_qwen_pt and med_qwen_pt + NTP   textbooks
python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medqa,medmcqa,pubmedqa --model /niutrans/NEUNLP/rjh/sft/naive_qwen2_7b_textbooks_7m28d_bsz256_cosine_lr3e-04_warmratio5e-02,/niutrans/NEUNLP/rjh/sft/naive_qwen_med_pt_medquad_7m29d_bsz256_cosine_lr2e-05_template_warmratio5e-02 --output_path  fuck

# pubmed abstract

python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset careqa --model /niutrans/NEUNLP/rjh/sft/naive_qwen2_7b_pubmed_abstract_7m29d_bsz256_cosine_lr3e-04_warmratio5e-02,/niutrans/NEUNLP/rjh/sftnaive_qwen_med_pt_alpaca_gpt4_medquad_7m30d_bsz256_cosine_lr2e-05_template_warmratio5e-02 --output_path  fuck



python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medqa,medmcqa,pubmedqa,bioasq,careqa,mmlu --model /niutrans/NEUNLP/rjh/sft/naive_qwen2_7b_alpaca_gpt4_medquad_7m31d_bsz256_cosine_lr2e-05_template_warmratio5e-02 --output_path  fuck


# BASE+NDP（womono）
qwen2_7b_alpaca_gpt4_medquad_8m1d_0_bsz256_alpha0.8_cosine_lr2e-05_mix0.8_template_warmratio5e-02

python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medqa,medmcqa,pubmedqa,bioasq,careqa,mmlu --model /niutrans/NEUNLP/rjh/sft/qwen2_7b_alpaca_gpt4_medquad_8m1d_0_bsz256_alpha0.8_cosine_lr2e-05_mix0.8_template_warmratio5e-02/checkpoint-267,/niutrans/NEUNLP/rjh/sft/qwen2_7b_alpaca_gpt4_medquad_8m1d_0_bsz256_alpha0.8_cosine_lr2e-05_mix0.8_template_warmratio5e-02/checkpoint-534,/niutrans/NEUNLP/rjh/sft/qwen2_7b_alpaca_gpt4_medquad_8m1d_0_bsz256_alpha0.8_cosine_lr2e-05_mix0.8_template_warmratio5e-02/checkpoint-801 --output_path  fuck


# BASE+NDP（mono）4gram

qwen2_7b_alpaca_gpt4_medquad_8m3d_0_bsz256_alpha0.8_cosine_lr2e-05_mix0.8_template_warmratio5e-02

python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medqa,medmcqa,pubmedqa,bioasq,careqa,mmlu --model /niutrans/NEUNLP/rjh/sft/qwen2_7b_alpaca_gpt4_medquad_8m3d_0_bsz256_alpha0.8_cosine_lr2e-05_mix0.8_template_warmratio5e-02/checkpoint-267,/niutrans/NEUNLP/rjh/sft/qwen2_7b_alpaca_gpt4_medquad_8m3d_0_bsz256_alpha0.8_cosine_lr2e-05_mix0.8_template_warmratio5e-02/checkpoint-534,/niutrans/NEUNLP/rjh/sft/qwen2_7b_alpaca_gpt4_medquad_8m3d_0_bsz256_alpha0.8_cosine_lr2e-05_mix0.8_template_warmratio5e-02/checkpoint-801 --output_path  fuck


# llama3 base+ntp
# naive_llama3_8b_medquad_alpaca_gpt4_8m4d_bsz256_cosine_lr5e-05_template_warmratio5e-02
python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medqa,medmcqa,pubmedqa,bioasq,careqa,mmlu --model /niutrans/NEUNLP/rjh/sft/naive_llama3_8b_medquad_alpaca_gpt4_8m4d_bsz256_cosine_lr5e-05_template_warmratio5e-02,/niutrans/NEUNLP/rjh/models/Llama-3-8B --output_path  fuck


python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medqa,medmcqa,pubmedqa,careqa,mmlu --model /niutrans/NEUNLP/rjh/sft/llama3_8b_alpaca_gpt4_medquad_8m5d_0_bsz256_alpha0.8_cosine_lr5e-05_mix0.8_template_warmratio5e-02/checkpoint-267,/niutrans/NEUNLP/rjh/sft/llama3_8b_alpaca_gpt4_medquad_8m5d_0_bsz256_alpha0.8_cosine_lr5e-05_mix0.8_template_warmratio5e-02/checkpoint-534,/niutrans/NEUNLP/rjh/sft/llama3_8b_alpaca_gpt4_medquad_8m5d_0_bsz256_alpha0.8_cosine_lr5e-05_mix0.8_template_warmratio5e-02/checkpoint-801 --output_path  fuck




python -m sft.eval.gsm8k  --mode 0 --shot --dp --dataset medqa,medmcqa,pubmedqa,careqa,mmlu --model /niutrans/NEUNLP/rjh/sft/llama3_8b_alpaca_gpt4_medquad_8m5d_0_bsz256_alpha0.8_cosine_lr5e-05_mix0.8_template_warmratio5e-02/checkpoint-267,/niutrans/NEUNLP/rjh/sft/llama3_8b_alpaca_gpt4_medquad_8m5d_0_bsz256_alpha0.8_cosine_lr5e-05_mix0.8_template_warmratio5e-02/checkpoint-534 --output_path  fuck

/data/lxy/llama_magpie_300k_template
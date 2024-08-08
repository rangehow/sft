
# 1.修改下面的字典，放入新建的config.py中，使其包含所有需要的本地模型和数据集
# model_dir = {
#     "llama3_8b": "",
# }
# dataset_dir = {
#     "magpie": "",
# }



# 2.完成数据集生成过程
python preprocess_trie.py --model llama3_8b --dataset magpie --template llama


# 3.开始训练，先训练我们的方法，然后是原始NTP
accelerate launch --config_file my.yaml special_train.py --model llama3_8b --gradient_accumulation_steps 32 --total_bsz 512 --zero_prob 0 --div_mode False --weighted False --dataset magpie --mix True --mix_ratio 0.8 --num_train_epochs 9

python special_train.py --model llama3_8b --gradient_accumulation_steps 128 --total_bsz 512 --zero_prob 0 --div_mode False --weighted False --dataset magpie --mix True --mix_ratio 0.8 --num_train_epochs 9



# accelerate launch --config_file my.yaml naive_train.py --model llama3_8b --dataset magpie --total_bsz 512  --gradient_accumulation_steps 32 --output_dir llama_naive_bsz512_mix  --num_train_epochs 9



python naive_train.py --model llama3_8b --dataset magpie --total_bsz 512  --gradient_accumulation_steps 64 --output_dir llama_naive_bsz512_mix  --num_train_epochs 3


# pretrain
accelerate launch --config_file megatron.yaml  naive_train.py --model llama3_8b --dataset wiki_medical --total_bsz 512  --gradient_accumulation_steps 64 --output_dir llama_med_pt  --num_train_epochs 1 --w_template False
# sft
accelerate launch --config_file megatron.yaml  naive_train.py naive_train.py --model llama3_8b --dataset medquad --total_bsz 512  --gradient_accumulation_steps 64 --output_dir llama_med_sft  --num_train_epochs 3 --w_template False


CUDA_VISIBLE_DEVICES=0,1 python naive_train.py --model llama3_8b --lora --dataset textbooks,wiki_medical,medical_transcription --total_bsz 512  --gradient_accumulation_steps 512 --output_dir llama_med_pt  --num_train_epochs 1 --w_template False


CUDA_VISIBLE_DEVICES=0,1 python naive_train.py --model /niutrans/NEUNLP/rjh/sft/llama_med_pt/checkpoint-263 --lora --dataset medquad --total_bsz 512  --gradient_accumulation_steps 256 --output_dir llama_med_sft  --num_train_epochs 3 --w_template False


python3 naive_train.py --num_train_epochs 2 --w_template False --dataset magpie --total_bsz 32 --gradient_accumulation_steps 8 --learning_rate 2e-5 --lr_scheduler_type cosine --num_train_epochs 2 --warmup_steps 100 --model llama3_8b --output_dir llama_magpie





 torchrun --nproc-per-node 2   naive_train.py     --model llama3_8b     --dataset alpaca_gpt4,code,math     --total_bsz 512     --gradient_accumulation_steps 256   --num_train_epochs 3    --w_template False  --lora


 python naive_train.py --model mistral_7b --gradient_accumulation_steps 128 --total_bsz 256  --dataset textbooks,wiki_medical,medical_transcription  -w_template False --num_train_epochs 1 --learning_rate 6e-4 --lr_scheduler_type constant_with_warmup --warmup_ratio 0.05

python naive_train.py --model qwen2_7b --gradient_accumulation_steps 128 --total_bsz 256  --dataset textbooks  --w_template False --num_train_epochs 1 --learning_rate 3e-4 --lr_scheduler_type cosine --warmup_ratio 0.05



python naive_train.py --model qwen2_7b --gradient_accumulation_steps 128 --total_bsz 256  --dataset medquad --w_template True --num_train_epochs 2 --learning_rate 6e-4 --lr_scheduler_type cosine --warmup_ratio 0.05


python naive_train.py --model qwen_med_pt --gradient_accumulation_steps 128 --total_bsz 256  --dataset pubmedqa --w_template True --num_train_epochs 2 --learning_rate 2e-5 --lr_scheduler_type cosine --warmup_ratio 0.05


python preprocess_trie.py --dataset medquad,alpaca_gpt4 --model llama3_8b --template llama --w_template True --mono True --mono_dataset pubmed_abstract  --ngram 1

# 728晚上2253没跑完的
python naive_train.py --model qwen_med_pt --gradient_accumulation_steps 128 --total_bsz 256  --dataset alpaca_gpt4,medquad --w_template True --num_train_epochs 3 --learning_rate 2e-5 --lr_scheduler_type cosine --warmup_ratio 0.05

python naive_train.py --model qwen2_7b --gradient_accumulation_steps 128 --total_bsz 256  --dataset medquad --w_template True --num_train_epochs 3 --learning_rate 2e-5 --lr_scheduler_type cosine --warmup_ratio 0.05
#========================================

python special_train.py --model qwen2_7b --gradient_accumulation_steps 128 --total_bsz 256  --dataset medquad --w_template True --num_train_epochs 3 --learning_rate 2e-5 --lr_scheduler_type cosine --warmup_ratio 0.05 --mono True --mono_dataset textbooks --mix True --mix_ratio 0.8 --div_mode False --weighted False --template qwen2


CUDA_VISIBLE_DEVICES=1 python special_train.py --model llama3_8b --gradient_accumulation_steps 512 --total_bsz 512  --dataset alpaca_gpt4,math,code --w_template True --num_train_epochs 3 --learning_rate 2e-5 --lr_scheduler_type cosine --warmup_ratio 0.05  --mix True --mix_ratio 0.8 --div_mode False --weighted False --template llama --lora



python preprocess_trie.py --dataset alpaca_gpt4,math,code --model llama3_8b --template llama --w_template True --ngram 1



python naive_train.py --model llama3_8b --gradient_accumulation_steps 256 --total_bsz 512  --dataset alpaca_gpt4,math,code --w_template True --num_train_epochs 3 --learning_rate 2e-5 --lr_scheduler_type cosine --warmup_ratio 0.05


python special_train.py --model qwen_med_pt --gradient_accumulation_steps 128 --total_bsz 256  --dataset medquad --w_template True --num_train_epochs 3 --learning_rate 2e-5 --lr_scheduler_type cosine --warmup_ratio 0.05 --mono True --mono_dataset textbooks --mix True --mix_ratio 0.8 --div_mode False --weighted False --template qwen2


python naive_train.py --model qwen2_7b --gradient_accumulation_steps 128 --total_bsz 256  --dataset pubmed_abstract  --w_template False --num_train_epochs 1 --learning_rate 3e-4 --lr_scheduler_type cosine --warmup_ratio 0.05



python naive_train.py --model llama3_8b --gradient_accumulation_steps 128 --total_bsz 256  --dataset magpie --w_template True --num_train_epochs 3 --learning_rate 5e-5 --lr_scheduler_type cosine --warmup_ratio 0.05


# 8/1
google/gemma-2-2b

python naive_train.py --model gemma2_2b --gradient_accumulation_steps 256 --total_bsz 256  --dataset magpie --w_template True --num_train_epochs 3 --learning_rate 5e-5 --lr_scheduler_type cosine --warmup_ratio 0.05


python special_train.py --model qwen2_7b --gradient_accumulation_steps 128 --total_bsz 256  --dataset medquad,alpaca_gpt4 --w_template True --num_train_epochs 3 --learning_rate 2e-5 --lr_scheduler_type cosine --warmup_ratio 0.05  --mix True --mix_ratio 0.8 --div_mode False --weighted False --template qwen2



# 8/4

python naive_train.py --model llama3_8b --gradient_accumulation_steps 128 --total_bsz 256  --dataset medquad,alpaca_gpt4 --w_template True --num_train_epochs 2 --learning_rate 5e-5 --lr_scheduler_type cosine --warmup_ratio 0.05

python special_train.py --model llama3_8b --gradient_accumulation_steps 128 --total_bsz 256  --dataset alpaca_gpt4,medquad --w_template True --num_train_epochs 2 --learning_rate 5e-5 --lr_scheduler_type cosine --warmup_ratio 0.05  --mix True --mix_ratio 0.8 --div_mode False --weighted False --template llama --mono True --mono_dataset pubmed_abstract


# 8/5
python preprocess_trie.py --model qwen2_7b --dataset alpaca_gpt4,medquad --template qwen2 --mono True --mono_dataset pubmed_abstract,redpajama --ngram 1

python preprocess_trie.py --model llama3_8b --dataset alpaca_gpt4,medquad --template llama --ngram 4 && python special_train.py --model llama3_8b --gradient_accumulation_steps 128 --total_bsz 256  --dataset alpaca_gpt4,medquad --w_template True --num_train_epochs 2 --learning_rate 5e-5 --lr_scheduler_type cosine --warmup_ratio 0.05  --mix True --mix_ratio 0.8 --div_mode False --weighted False --template llama 

# 8/6

这个
python special_train.py --model qwen2_7b --gradient_accumulation_steps 128 --total_bsz 256  --dataset alpaca_gpt4,medquad --w_template True --num_train_epochs 2 --learning_rate 2e-5 --lr_scheduler_type cosine --warmup_ratio 0.05  --mix True --mix_ratio 0.8 --div_mode False --weighted False --template qwen2 --mono True --mono_dataset pubmed_abstract,redpajama


# 8/8 知识蒸馏

CUDA_VISIBLE_DEVICES=2,3 python distillation_train.py --model gemma_2b --teacher_model gemma2_27b_it --total_bsz 512 --gradient_accumulation_steps 256 --dataset alpaca_gpt4,code,math
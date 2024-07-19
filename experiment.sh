
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


accelerate launch --config_file my.yaml special_train.py  --model llama3_8b --gradient_accumulation_steps 32 --total_bsz 256 --zero_prob 0 --alpha 1 --div_mode False &



CUDA_VISIBLE_DEVICES=2 python -m sft.slide_train --gradient_accumulation_steps 64 --total_bsz 128 --zero_prob 0 --div_mode False --dataset /niutrans/NEUNLP/rjh/sft/ndp/train_dataset/qwen2.5-1.5b/qwen2.5_alma_zhen/weighted/w8_b0.5 --learning_rate 5e-5 --template qwen2.5  --lr_scheduler_type cosine --warmup_ratio 0.01 --model qwen2.5-1.5b --w_template True



CUDA_VISIBLE_DEVICES=2 python naive_train.py --gradient_accumulation_steps 64 --total_bsz 128 --zero_prob 0 --div_mode False --dataset alma_zhen --learning_rate 5e-5 --template qwen2.5  --lr_scheduler_type cosine --warmup_ratio 0.01 --model qwen2.5-1.5b --w_template True
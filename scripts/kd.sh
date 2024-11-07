STUDENT_MODEL_DIR=
TEACHER_MODEL_DIR=

python naive_train.py --dataset alpaca_gpt4,code,math --gradient_accumulation_steps 512 --total_bsz 512 --model gemma_2b --teacher_model   gemma2_27b_it --num_train_epochs 3 --template gemma



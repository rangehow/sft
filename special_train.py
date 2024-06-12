import ast
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    BartTokenizerFast,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    BartTokenizer,
)
from torch.utils.data import Dataset, DataLoader
import datasets
from dataset import SpecialDataset, SpecialDataCollator
from special_trainer import KLTrainer
import pickle
from config import model_dir, dataset_dir
import torch
from argparse import ArgumentParser
from loguru import logger
import warnings
import os


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", default="gemma_2b")
    parser.add_argument("--dataset", default="alpaca_gpt4")
    parser.add_argument("--div_mode", default=True, type=ast.literal_eval)
    parser.add_argument("--output_dir")
    parser.add_argument("--fa2", action="store_true", help="decide to use fa2 or not")
    parser.add_argument("--lora", action="store_true", help="decide to use lora or not")
    parser.add_argument("--zero_prob", default=0.1, type=ast.literal_eval)
    parser.add_argument("--gradient_accumulation_steps", default=8, type=int)
    parser.add_argument("--total_bsz", default=64, type=int)
    parser.add_argument(
        "--weighted", action="store_true", help="decide to use token level freq weight"
    )
    return parser.parse_args()


args = parse_args()


if args.output_dir is None:
    from datetime import datetime

    current_time = datetime.now()
    current_month = current_time.month
    current_day = current_time.day
    args.output_dir = f"{args.model}_{args.dataset}_{current_month}m{current_day}d_{args.zero_prob}_bsz{args.total_bsz}"
    if args.weighted:
        args.output_dir = args.output_dir + "_weighted"
    if args.div_mode:
        args.output_dir = args.output_dir + "_div"
    if args.lora:
        args.output_dir = args.output_dir + "_lora"
    logger.info(f"未检测到output_dir，故采用自动生成的{args.output_dir}")

model_dir = model_dir.get(args.model, args.model)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto",
    # device_map="auto",  # 在显存不够的时候优先考虑流水线并行吧。 这样不需要考虑变化的总bsz
    attn_implementation="flash_attention_2" if args.fa2 else "sdpa",
)
model_type = model.config.model_type
embedding_size = model.lm_head.weight.size()[
    0
]  # 取lm_head比较安全，因为有些模型embedding layer会取不同的名字

collator = SpecialDataCollator(
    tokenizer,
    zero_prob=args.zero_prob,
    embedding_size=embedding_size,
    div_mode=args.div_mode,
)


@logger.catch
def load_dataset():
    script_path = os.path.dirname(os.path.abspath(__file__).rstrip(os.sep))
    with open(
        f"{script_path}/train_dataset/{model_type}_{args.dataset}_synthesis.pkl", "rb"
    ) as f:
        synthesis = pickle.load(f)

    with open(
        f"{script_path}/train_dataset/{model_type}_{args.dataset}_index.pkl", "rb"
    ) as f:
        index = pickle.load(f)

    train_dataset = SpecialDataset(
        synthesis,
        index,
        embedding_size,
        div_mode=args.div_mode,
    )
    return train_dataset


train_dataset = load_dataset()
# 检查数据的调试代码----------------------------------
# dataloader = DataLoader(
#     dataset=train_dataset, batch_size=8, collate_fn=collator, num_workers=16,pin_memory=True
# )

# from tqdm import tqdm


# for d in tqdm(dataloader):
#     continue

# ------------------------------------------------------
logger.debug(f"训练集大小：{len(train_dataset)}")
logger.debug(args)

real_bsz = args.total_bsz // torch.cuda.device_count() // 8
logger.debug(
    f"实际的总batch_size=梯度累计{args.gradient_accumulation_steps}x每张卡的bsz{real_bsz}x卡的数量{torch.cuda.device_count()}={args.gradient_accumulation_steps*real_bsz*torch.cuda.device_count()}"
)

if args.lora:
    from peft import LoraConfig, TaskType, get_peft_model

    peft_config = LoraConfig(
        inference_mode=False,
        r=64,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules="all-linear",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()


# torch.backends.cudnn.benchmark = False
trainer = KLTrainer(
    weight_mode=args.weighted,
    model=model,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    args=TrainingArguments(
        overwrite_output_dir=True,
        output_dir=args.output_dir,
        logging_steps=1,
        remove_unused_columns=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_strategy="epoch",
        dataloader_pin_memory=False,
        dataloader_num_workers=12,
        num_train_epochs=3,
        per_device_train_batch_size=real_bsz,
        bf16=True,
    ),
    data_collator=collator,
)

# from tqdm import tqdm

# d = trainer.get_train_dataloader()
# for dd in tqdm(d):
#     continue


trainer.train()
trainer.save_model(args.output_dir)


saved_args_dict = vars(args)
saved_args_dict["实际的总batch_size"] = (
    args.gradient_accumulation_steps * real_bsz * torch.cuda.device_count()
)
with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as o:
    json.dump(saved_args_dict, o, ensure_ascii=False, indent=4)

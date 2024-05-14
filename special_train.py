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
# import torch.multiprocessing as mp
# mp.set_start_method('spawn')



def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", default="gemma_2b")
    parser.add_argument("--dataset", default="alpaca_cleaned")
    parser.add_argument("--div_mode", default=False, type=ast.literal_eval)
    parser.add_argument("--output_dir")
    parser.add_argument("--fa2", action="store_true", help="decide to use fa2 or not")
    return parser.parse_args()


args = parse_args()


model_dir = model_dir[args.model]
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2" if args.fa2 else "sdpa",
)

embedding_size = model.lm_head.weight.size()[
    0
]  # 取lm_head比较安全，因为有些模型embedding layer会取不同的名字

collator = SpecialDataCollator(tokenizer,zero_prob=0.1,embedding_size=embedding_size)


@logger.catch
def load_dataset():
    with open(f"{dataset_dir[args.dataset]}/synthesis.pkl", "rb") as f:
        synthesis = pickle.load(f)

    with open(f"{dataset_dir[args.dataset]}/index.pkl", "rb") as f:
        index = pickle.load(f)

    train_dataset = SpecialDataset(
        synthesis,
        index,
        embedding_size,
        zero_prob=0.1,
        div_mode=args.div_mode,
    )
    return train_dataset


train_dataset = load_dataset()
# 检查数据的调试代码----------------------------------
# dataloader = DataLoader(
#     dataset=train_dataset, batch_size=8, collate_fn=collator, num_workers=30,pin_memory=True
# )

# from tqdm import tqdm


# for d in tqdm(dataloader):
#     del d
# ------------------------------------------------------


trainer = KLTrainer(
    model=model,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    args=TrainingArguments(
        optim="adamw_apex_fused",
        overwrite_output_dir=False,
        output_dir=args.output_dir,
        logging_steps=1,
        remove_unused_columns=False,
        gradient_accumulation_steps=8,
        save_strategy="epoch",
        dataloader_pin_memory =False,
        dataloader_num_workers=0,
        num_train_epochs=3,
        auto_find_batch_size=True,
        bf16=True,
    ),
    data_collator=collator,
)

trainer.train()
trainer.save_model(args.output_dir)

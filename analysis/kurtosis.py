from functools import partial
import json
import sys
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    Trainer,
    TrainingArguments,
    DataCollator,
)
from ..config import *
from ..template import modelType2Template
from argparse import ArgumentParser
import datasets
from loguru import logger
from ..dataset_func import dname2func
from ..eval.load_func import dname2load
import torch
import ast
import os
from ..model_utils import balanced_load
import numpy as np



class MyCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, examples) -> torch.Any:
        input_ids = [list(example["input_ids"][:8192]) for example in examples]
        labels = [list(example["labels"][:8192]) for example in examples]

        input_ids_padded = self.tokenizer.pad(
            {"input_ids": input_ids},
            return_tensors="pt",
            padding=True,
        )
        max_len = input_ids_padded.input_ids.shape[-1]
        labels_padded = torch.tensor(
            [[-100] * (max_len - len(label)) + label for label in labels]
        )
        return {
            "input_ids": input_ids_padded.input_ids,
            "attention_mask": input_ids_padded.attention_mask,
            "labels": labels_padded,
        }


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--dataset", default="alpaca_gpt4,math,code")
    parser.add_argument(
        "--output_path",
    )
    return parser.parse_args()


args = parse_args()


model_dir = model_dir.get(args.model, args.model)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# model=balanced_load(model_dir,num_devices=torch.cuda.device_count())
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto",
    device_map="auto",
    low_cpu_mem_usage=True,
    # attn_implementation="eager" if 'gemma2' in args.model else 'sdpa',
)

# NOTE 从config.json中读取模型的类型，从而自动获取合适的模板类型
# config = AutoConfig.from_pretrained(model_dir)
model_type = model.config.model_type
template = modelType2Template[model_type](tokenizer)

my_collator = MyCollator(tokenizer)
# 读取数据集
dataset_name_list = args.dataset.split(",")
dataset_list = []
for dname in dataset_name_list:
    train_dataset = dname2load[dname](dataset_dir.get(args.dataset, None))
    print(train_dataset)
    train_dataset = train_dataset.map(
        partial(
            dname2func[dname],
            template=template,
            mode=1,
            test=False,
        ),
        batched=True,
        num_proc=60,
        remove_columns=train_dataset.features.keys(),
        # load_from_cache_file=False,
        desc="tokenize",
    )
    dataset_list.append(train_dataset)
    
    
    
def merge_dicts(dataset_list):
    # Initialize a new dictionary to store the merged data
    merged_dict = {'input_ids': [] ,'labels':[]}
    
    # Define how many items to take from each dictionary
    num_items = [8000, 1000, 1000]
    
    for idx, data_dict in enumerate(dataset_list):
        for key in ['input_ids','labels']:
            # Take the specified number of items from each dictionary
            merged_dict[key].extend(data_dict[key][:num_items[idx]])
            
    return merged_dict



train_dataset = datasets.Dataset.from_dict(merge_dicts(dataset_list))

input_ids = train_dataset[0]["input_ids"]
labels = train_dataset[0]["labels"]

if -100 in labels:
    filtered_tensor = labels[labels.index(-100) + labels.count(-100) :]
else:
    filtered_tensor = labels
logger.debug("input_ids")
print(tokenizer.decode(input_ids))
print(tokenizer.convert_ids_to_tokens(input_ids))
logger.debug("labels")
print(tokenizer.convert_ids_to_tokens(filtered_tensor))

logger.debug(f"数据集总量是{len(train_dataset)}")

from torch.utils.data import DataLoader

dataloader = DataLoader(
    train_dataset,
    collate_fn=my_collator,
    batch_size=2,
    num_workers=16,
    pin_memory=True,
)
from tqdm import tqdm
import torch


def kurtosis_pytorch(data, dim=-1):
    # Convert data to a PyTorch tensor if not already
    data_tensor = data if isinstance(data, torch.Tensor) else torch.tensor(data, dtype=torch.float32)

    # Calculate mean and standard deviation without Bessel's correction
    mean = torch.mean(data_tensor, dim=dim, keepdim=True)
    std = torch.std(data_tensor, dim=dim, unbiased=False, keepdim=True)



    # Calculate kurtosis
    n = data_tensor.size(dim)
    kurtosis_value = torch.sum(((data_tensor - mean) / (std + 1e-12)) ** 4, dim=dim) / n
    
    # Adjust for normal distribution kurtosis
    kurtosis_value = kurtosis_value - 3
    
    return kurtosis_value

kurtosis_values = 0
for instance in tqdm(dataloader):
    input_ids = instance["input_ids"].to(model.device)
    attention_mask = instance["attention_mask"].to(model.device)
    labels = instance["labels"].to(model.device)
    # print(attention_mask.device)
    with torch.no_grad():
       
        logit = model(input_ids=input_ids, attention_mask=attention_mask).logits[
            labels != -100
        ]

        kurtosis_values += torch.sum(kurtosis_pytorch(logit)).item()

print("kurtosis=", kurtosis_values)
config_str = f"{args.model}"
result = {
    "kurtosis_values": kurtosis_values,
}
output_path = args.output_path
if os.path.exists(output_path):
    with open(output_path, "r") as f:
        results = json.load(f)
else:
    results = {}

    # 以配置字符串为键，结果为值保存
results[config_str] = result

with open(output_path, "w") as f:
    json.dump(results, f, indent=4)

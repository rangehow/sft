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
    return parser.parse_args()


args = parse_args()
from transformers import AutoModelForCausalLM, AutoConfig


# model1 = AutoModelForCausalLM.from_pretrained(args.model)
# model2 = AutoModelForCausalLM.from_pretrained("/niutrans/NEUNLP/rjh/models/Llama-3-8B")

# state_dict1 = model1.state_dict()
# state_dict2 = model2.state_dict()


# def compare_models(state_dict1, state_dict2):
#     if state_dict1.keys() != state_dict2.keys():
#         print("模型参数键不匹配")
#         return False

#     for key in state_dict1.keys():
#         if not torch.equal(state_dict1[key], state_dict2[key]):
#             print(f"参数 {key} 不一致")
#             # return False
#         else:
#             print(f"参数 {key} 一致aaaa")


# # print(state_dict1.keys())
# # print("-" * 20)
# # print(state_dict2.keys())
# # 执行比较

# compare_models(state_dict1, state_dict2)
# exit()
from peft import PeftModelForCausalLM

with open(os.path.join(args.model, "adapter_config.json")) as o:
    base_model = json.load(o)["base_model_name_or_path"]

print(base_model, args.model)
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.save_pretrained(args.model)
base_model = AutoModelForCausalLM.from_pretrained(base_model)
model = PeftModelForCausalLM.from_pretrained(base_model, args.model)
merged_model = model.merge_and_unload()
merged_model.save_pretrained(args.model)

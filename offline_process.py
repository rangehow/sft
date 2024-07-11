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
import multiprocessing
from tqdm import tqdm
from dataset import directly_softmax
from preprocess_trie import save_chunks


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--div_mode", default=True, type=ast.literal_eval)
    parser.add_argument("--output_dir")
    parser.add_argument("--alpha", default=0.8, type=ast.literal_eval)
    parser.add_argument("--fa2", action="store_true", help="decide to use fa2 or not")
    parser.add_argument("--lora", action="store_true", help="decide to use lora or not")
    parser.add_argument("--zero_prob", default=0.1, type=ast.literal_eval)
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
    parser.add_argument("--total_bsz", default=128, type=int)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument(
        "--weighted",
        default=True,
        type=ast.literal_eval,
        help="decide to use token level freq weight",
    )
    parser.add_argument(
        "--mix",
        default=False,
        type=ast.literal_eval,
        help="decide to use token level freq weight",
    )
    parser.add_argument(
        "--mix_ratio", default=0.8, type=ast.literal_eval, help="sft信号的融合比例"
    )
    parser.add_argument(
        "--pt",
        default=False,
        type=ast.literal_eval,
        help="pt mode or not?",
    )
    parser.add_argument(
        "--template",
    )
    return parser.parse_args()


args = parse_args()


if args.output_dir is None:
    from datetime import datetime

    current_time = datetime.now()
    current_month = current_time.month
    current_day = current_time.day
    args.output_dir = f"{args.model}_{args.dataset.replace(',','_')}_{current_month}m{current_day}d_{args.zero_prob}_bsz{args.total_bsz}_alpha{args.alpha}"
    if args.weighted:
        args.output_dir = args.output_dir + "_weighted"
    if args.div_mode:
        args.output_dir = args.output_dir + "_div"
    if args.mix:
        args.output_dir = args.output_dir + f"_mix{args.mix_ratio}"
    if args.lora:
        args.output_dir = args.output_dir + "_lora"
    logger.info(f"未检测到output_dir，故采用自动生成的{args.output_dir}")

model_dir = model_dir.get(args.model, args.model)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
logger.debug(f"模型路径是:{model_dir}")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto",
    low_cpu_mem_usage=True,
    # device_map="balanced_low_0",  # 在显存不够的时候优先考虑流水线并行吧。 这样不需要考虑变化的总bsz
)

embedding_size = model.lm_head.weight.size()[
    0
]  # 取lm_head比较安全，因为有些模型embedding layer会取不同的名字

collator = SpecialDataCollator(
    tokenizer,
    zero_prob=args.zero_prob,
    embedding_size=embedding_size,
    div_mode=args.div_mode,
    mix=args.mix,
    mix_ratio=args.mix_ratio,
    pt=args.pt,
    offline=False,
)


def load_msgpack_file(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)  # ,strict_map_key=False,strict_types =True


def find_msgpack_chunk_files(
    base_dir,
    name,
):
    """查找与基准文件名匹配的所有 msgpack 分块文件。"""

    chunk_files = [
        os.path.join(base_dir, f)
        for f in os.listdir(base_dir)
        if f.startswith(name) and f.endswith(".msgpack")
    ]
    return sorted(chunk_files)


import concurrent.futures


def load_msgpack_chunks(chunk_files):

    print(chunk_files)
    # cpu_count = multiprocessing.cpu_count()
    # logger.debug(f"加载数据集使用CPU 核心数：{cpu_count//2}")  cpu_count // 2
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(load_msgpack_file, chunk_files),
                total=len(chunk_files),
                desc="loading files",
            )
        )
    if isinstance(results[0], dict):
        merged_data = {}
        for chunk in results:
            merged_data.update(chunk)
        return merged_data
    elif isinstance(results[0], list):
        merged_data = []
        for chunk in results:
            merged_data.extend(chunk)
        return merged_data
    else:
        raise TypeError("data must be a dictionary or a list")


@logger.catch
def load_dataset():
    script_path = os.path.dirname(os.path.abspath(__file__).rstrip(os.sep))
    # with open(
    #     f"{script_path}/train_dataset/{model_type}_{args.dataset}_synthesis.pkl", "rb"
    # ) as f:
    #     synthesis = pickle.load(f)

    # with open(
    #     f"{script_path}/train_dataset/{model_type}_{args.dataset}_index.pkl", "rb"
    # ) as f:
    #     index = pickle.load(f)

    base_dir = f"{script_path}/train_dataset/{args.template}_{args.dataset}"

    synthesis = load_msgpack_chunks(
        find_msgpack_chunk_files(base_dir, name="synthesis")
    )
    index = load_msgpack_chunks(find_msgpack_chunk_files(base_dir, name="index"))
    train_dataset = SpecialDataset(
        synthesis,
        index,
        embedding_size,
        zero_prob=args.zero_prob,
        div_mode=args.div_mode,
        pt=args.pt,
    )
    return train_dataset


train_dataset = load_dataset()

dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=1,  # 必须是1,不然要从pad里解离出各种东西太麻烦。
    collate_fn=collator,
    num_workers=16,
    pin_memory=False,
)

from tqdm import tqdm

total_data = []
idx = 0
script_path = os.path.dirname(os.path.abspath(__file__).rstrip(os.sep))
os.makedirs(
    os.path.join(
        script_path, "train_dataset", f"{args.template}_{args.dataset}_offline"
    ),
    exist_ok=True,
)
chunk_size = 256
for d in tqdm(dataloader):

    # 1.只有input_ids需要还原成list，因为不同话的不齐，但是其他都是在time上拼接的
    if args.mix:
        result = {
            "input_ids": d["input_ids"].tolist()[0],
            "all_prob_mix": d["all_prob_mix"],
            "mix_cnt": d["mix_cnt"],
            "valid_label_index_list": d["valid_label_index_list"][0],
        }
    elif args.pt:
        result = {
            "input_ids": d["input_ids"].tolist()[0],
            "all_prob_clm": d["all_prob_clm"],
            "clm_cnt": d["clm_cnt"],
            "valid_label_index_list": d["valid_label_index_list"][0],
        }
    else:
        result = {
            "input_ids": d["input_ids"].tolist()[0],
            "all_prob_supervised": d["all_prob_supervised"],
            "all_prob_clm": d["all_prob_clm"],
            "supervised_cnt": d["supervised_cnt"],
            "clm_cnt": d["clm_cnt"],
            "valid_label_index_list": d["valid_label_index_list"][0],
        }

    total_data.append(result)
    if len(total_data) == chunk_size:
        save_chunks(
            total_data,
            chunk_size=chunk_size,
            base_dir=f"{script_path}/train_dataset/{args.template}_{args.dataset}_offline",
            name="synthesis",
            start_idx=idx,
        )
        idx += 1
        total_data = []


save_chunks(
    total_data,
    chunk_size=chunk_size,
    base_dir=f"{script_path}/train_dataset/{args.template}_{args.dataset}_offline",
    name="synthesis",
    start_idx=idx,
)


# ------------------------------------------------------

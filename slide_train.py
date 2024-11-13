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
from .dataset import SlideDataset, SlideDataCollator
from .ndp_trainer import NDPTrainer
import pickle
from .config import model_dir, dataset_dir
import torch
from argparse import ArgumentParser
from loguru import logger
import warnings
import os
import multiprocessing
from tqdm import tqdm
from .shm_utils import get_shm_info
from ipdb import set_trace as bp
from multiprocessing import Pool
from niuload import balanced_load


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--dataset", help="local dir to ndp process directory")
    parser.add_argument("--div_mode", default=False, type=ast.literal_eval)
    parser.add_argument("--output_dir")
    parser.add_argument("--lora", action="store_true", help="decide to use lora or not")
    parser.add_argument("--zero_prob", default=0, type=ast.literal_eval)
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
    parser.add_argument("--total_bsz", default=128, type=int)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument(
        "--template",
    )

    parser.add_argument(
        "--pt",
        default=False,
        type=ast.literal_eval,
        help="pt mode or not?",
    )

    parser.add_argument(
        "--mono",
        default=False,
        type=ast.literal_eval,
    )
    parser.add_argument(
        "--w_template",
        default=False,
        type=ast.literal_eval,
    )
    parser.add_argument("--mono_dataset")
    parser.add_argument("--warmup_ratio", type=float, default=0)
    parser.add_argument("--lr_scheduler_type", default="linear")
    parser.add_argument("--learning_rate", default=5e-5, type=ast.literal_eval)
    return parser.parse_args()


args = parse_args()


def is_torchrun():
    # torchrun 通常会设置 RANK 和 WORLD_SIZE 环境变量
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


if is_torchrun():
    real_bsz = (
        args.total_bsz // args.gradient_accumulation_steps // torch.cuda.device_count()
    )
    logger.debug(f"data parallel mode")
    logger.debug(
        f"实际的总batch_size=梯度累计{args.gradient_accumulation_steps}x每张卡的bsz{real_bsz} x 卡数{torch.cuda.device_count()} ={args.gradient_accumulation_steps*real_bsz*torch.cuda.device_count()}"
    )
else:
    real_bsz = args.total_bsz // args.gradient_accumulation_steps
    logger.debug(f"pipeline parallel mode")
    logger.debug(
        f"实际的总batch_size=梯度累计{args.gradient_accumulation_steps}x bsz{real_bsz}={args.gradient_accumulation_steps*real_bsz}"
    )


def load_single_chunk(args):
    base_dir, name, i = args
    filename = f"{name}_part{i}.pkl"
    with open(os.path.join(base_dir, filename), "rb") as f:
        chunk = pickle.load(f)
    return chunk


def load_chunks_parallel(base_dir, name, num_processes=None):
    # 读取元数据
    with open(os.path.join(base_dir, f"{name}_metadata.json")) as f:
        metadata = json.load(f)

    # 准备进程池参数
    args = [(base_dir, name, i) for i in range(metadata["num_chunks"])]

    # 使用进程池并行加载
    with Pool(processes=num_processes) as pool:
        chunks = pool.map(load_single_chunk, args)

    # 合并所有列表数据
    # 合并chunks
    combined_data = {}
    # 检查第一个chunk的类型来确定原始数据类型
    if isinstance(chunks[0], dict):
        # 如果是字典，将所有chunks合并成一个字典
        for chunk in chunks:
            combined_data.update(chunk)
    elif isinstance(chunks[0], list):
        # 如果是列表，将所有chunks串联成一个列表
        combined_data = []
        for chunk in chunks:
            combined_data.extend(chunk)
    else:
        raise TypeError("Unsupported data type in chunks")
    return combined_data


@logger.catch
def load_dataset():

    synthesis = load_chunks_parallel(args.dataset, name="synthesis")
    index = load_chunks_parallel(args.dataset, name="index")

    train_dataset = SlideDataset(
        synthesis,
        index,
        embedding_size=160000,
        zero_prob=args.zero_prob,
        div_mode=args.div_mode,
    )
    return train_dataset


model_dir = model_dir.get(args.model, args.model)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
logger.debug(f"模型路径是:{model_dir}")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


# 检查数据的调试代码----------------------------------
train_dataset = load_dataset()
collator = SlideDataCollator(
    tokenizer,
    zero_prob=args.zero_prob,
    embedding_size=160000,
    div_mode=args.div_mode,
)


dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=8,
    collate_fn=collator,
    num_workers=0,
    pin_memory=False,
)

from tqdm import tqdm


for d in tqdm(dataloader):
    bp()


model = balanced_load(model_dir)
# model = AutoModelForCausalLM.from_pretrained(
#     model_dir,
#     torch_dtype="auto",
#     device_map=(
#         "auto" if not is_torchrun() else None
#     ),  # 在显存不够的时候优先考虑流水线并行吧。 这样不需要考虑变化的总bsz
#     attn_implementation="flash_attention_2" if args.fa2 else "sdpa",
# )

embedding_size = model.lm_head.weight.size()[
    0
]  # 取lm_head比较安全，因为有些模型embedding layer会取不同的名字
train_dataset = load_dataset()

collator = SlideDataCollator(
    tokenizer,
    zero_prob=args.zero_prob,
    embedding_size=embedding_size,
    div_mode=args.div_mode,
    mix=args.mix,
    mix_ratio=args.mix_ratio,
    pt=args.pt,
)


# 检查数据的调试代码----------------------------------
# dataloader = DataLoader(
#     dataset=train_dataset,
#     batch_size=8,
#     collate_fn=collator,
#     num_workers=0,
#     pin_memory=False,
# )

# from tqdm import tqdm


# for d in tqdm(dataloader):
#     continue

# ------------------------------------------------------
logger.debug(f"训练集大小：{len(train_dataset)}")
logger.debug(args)

# real_bsz = (
#     args.total_bsz // torch.cuda.device_count() // args.gradient_accumulation_steps
# )


if args.lora:
    from peft import LoraConfig, TaskType, get_peft_model

    peft_config = LoraConfig(
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "embed_tokens",
            "lm_head",
            "layers.*.self_attn.q_proj",
            "layers.*.self_attn.k_proj",
            "layers.*.self_attn.v_proj",
            "layers.*.self_attn.o_proj",
            "layers.*.mlp.gate_proj",
            "layers.*.mlp.up_proj",
            "layers.*.mlp.down_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()


def get_output_dir(args):
    """
    Generate output directory path based on training arguments.
    """
    from datetime import datetime

    if args.output_dir is not None:
        return args.output_dir

    # Basic components
    current_time = datetime.now()
    base_name = f"{args.model}_{args.dataset.replace(',','_')}"
    date_str = f"{current_time.month}m{current_time.day}d"

    # Core training parameters
    train_params = [
        f"zp{args.zero_prob}",
        f"bsz{args.total_bsz}",
        f"{args.lr_scheduler_type}",
        f"lr{args.learning_rate:.0e}",
    ]

    # Optional components based on flags
    optional_components = []
    if args.div_mode:
        optional_components.append("div")
    if args.lora:
        optional_components.append("lora")
    if args.w_template:
        optional_components.append("template")
    if args.warmup_ratio > 0:
        optional_components.append(f"warmratio{args.warmup_ratio:.0e}")
    if args.pt:
        optional_components.append("pt")
    if args.mono:
        optional_components.append("mono")

    # Combine all components
    components = [base_name, date_str] + train_params + optional_components
    output_dir = "_".join(components)

    print(f"Auto-generated output directory: {output_dir}")
    return output_dir


# torch.backends.cudnn.benchmark = False
trainer = NDPTrainer(
    model=model,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    args=TrainingArguments(
        overwrite_output_dir=True,
        output_dir=args.output_dir if args.output_dir else get_output_dir(args),
        logging_steps=1,
        remove_unused_columns=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_strategy="epoch",
        dataloader_pin_memory=True,
        dataloader_num_workers=get_shm_info() // 30,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=real_bsz,
        bf16=True,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
    ),
    data_collator=collator,
)

# from tqdm import tqdm

# d = trainer.get_train_dataloader()
# for dd in tqdm(d):
#     continue


trainer.train()
trainer.save_model(args.output_dir)
trainer.save_state()

saved_args_dict = vars(args)
saved_args_dict["实际的总batch_size"] = (
    args.gradient_accumulation_steps * real_bsz * torch.cuda.device_count()
)
logger.info(f"模型被保存至{args.output_dir}")
with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as o:
    json.dump(saved_args_dict, o, ensure_ascii=False, indent=4)

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
from dataset import SpecialDataset, SpecialDataCollator, OfflineDataset
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
from shm_utils import get_shm_info


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--div_mode", default=False, type=ast.literal_eval)
    parser.add_argument("--output_dir")
    parser.add_argument("--alpha", default=0.8, type=ast.literal_eval)
    parser.add_argument("--fa2", action="store_true", help="decide to use fa2 or not")
    parser.add_argument("--lora", action="store_true", help="decide to use lora or not")
    parser.add_argument("--zero_prob", default=0, type=ast.literal_eval)
    parser.add_argument("--gradient_accumulation_steps", default=16, type=int)
    parser.add_argument("--total_bsz", default=128, type=int)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument(
        "--template",
    )
    parser.add_argument(
        "--weighted",
        default=False,
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
        "--offline",
        default=False,
        type=ast.literal_eval,
        help="offline mode or not?",
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

if args.output_dir is None:
    from datetime import datetime

    current_time = datetime.now()
    current_month = current_time.month
    current_day = current_time.day
    args.output_dir = f"{args.model}_{args.dataset.replace(',','_')}_{current_month}m{current_day}d_{args.zero_prob}_bsz{args.total_bsz}_alpha{args.alpha}_{args.lr_scheduler_type}_lr{args.learning_rate:.0e}"
    if args.weighted:
        args.output_dir = args.output_dir + "_weighted"
    if args.div_mode:
        args.output_dir = args.output_dir + "_div"
    if args.mix:
        args.output_dir = args.output_dir + f"_mix{args.mix_ratio}"
    if args.lora:
        args.output_dir = args.output_dir + "_lora"
    if args.w_template:
        args.output_dir = args.output_dir + "_template"
    if args.warmup_ratio > 0:
        args.output_dir = args.output_dir + f"_warmratio{args.warmup_ratio:.0e}"
    logger.info(f"未检测到output_dir，故采用自动生成的{args.output_dir}")

model_dir = model_dir.get(args.model, args.model)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
logger.debug(f"模型路径是:{model_dir}")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

from niuload import balanced_load
model = balanced_load(model_dir,ratio=[0.5,1,1,1])
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
    if args.offline:
        base_dir = base_dir + "_offline"
    if args.mono:
        base_dir = base_dir + f"_mono_{args.mono_dataset.replace(',','_')}"
    if args.w_template:
        base_dir = base_dir + f"_template"
    synthesis = load_msgpack_chunks(
        find_msgpack_chunk_files(base_dir, name="synthesis")
    )
    if not args.offline:
        index = load_msgpack_chunks(find_msgpack_chunk_files(base_dir, name="index"))

    if args.offline:
        train_dataset = OfflineDataset(synthesis)
    else:
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

collator = SpecialDataCollator(
    tokenizer,
    zero_prob=args.zero_prob,
    embedding_size=embedding_size,
    div_mode=args.div_mode,
    mix=args.mix,
    mix_ratio=args.mix_ratio,
    pt=args.pt,
    offline=args.offline,
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


# torch.backends.cudnn.benchmark = False
trainer = KLTrainer(
    pt_mode=args.pt,
    weight_mode=args.weighted,
    mix_mode=args.mix,
    alpha=args.alpha,
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

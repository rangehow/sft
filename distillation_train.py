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
from config import *
from template import modelType2Template
from argparse import ArgumentParser
import datasets
from loguru import logger
from dataset_func import dname2func
from eval.load_func import dname2load
import torch
import ast
import os
from kd_trainer import KDTrainer
import accelerate
from accelerate import infer_auto_device_map
from model_utils import balanced_load


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
    parser.add_argument("--model", default="gemma_2b")
    parser.add_argument("--dataset", default="alpaca_cleaned")

    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--lora", action="store_true", help="decide to use lora or not")
    parser.add_argument("--total_bsz", default=64, type=int)
    parser.add_argument("--label_smoothing_factor", default=0, type=float)
    parser.add_argument("--w_template", default=True, type=ast.literal_eval)
    parser.add_argument("--teacher_model", default=None)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0)
    parser.add_argument("--lr_scheduler_type", default="linear")
    parser.add_argument("--learning_rate", default=5e-5, type=ast.literal_eval)
    return parser.parse_args()


args = parse_args()


def is_torchrun():
    """
    检查是否通过 torchrun 启动。

    通过检查命令行参数，判断是否包含 torchrun 相关的参数。

    返回值：
    - 如果命令行参数包含 torchrun 相关的参数，返回 True。
    - 否则返回 False。
    """
    # 检查命令行参数中是否包含 torchrun 相关的参数
    args = sys.argv
    return "torchrun" in args or any(arg.startswith("--nproc_per_node") for arg in args)


def is_accelerate():
    """
    检查是否通过 torchrun 启动。

    通过检查命令行参数，判断是否包含 torchrun 相关的参数。

    返回值：
    - 如果命令行参数包含 torchrun 相关的参数，返回 True。
    - 否则返回 False。
    """
    # 检查命令行参数中是否包含 torchrun 相关的参数
    args = sys.argv
    return "accelerate" in args


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


student_model_dir = model_dir.get(args.model, args.model)
tokenizer = AutoTokenizer.from_pretrained(student_model_dir)
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


model = balanced_load(model_dir=student_model_dir, num_devices=2)

# model(torch.tensor([[1,2,3]]).to(model.device))

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
    # train_dataset = datasets.load_dataset(dataset_dir.get(args.dataset, args.dataset))[
    #     "train"
    # ]

    print(train_dataset)
    train_dataset = train_dataset.map(
        partial(
            dname2func[dname],
            template=template,
            mode=1 if args.w_template else 0,
            test=False,
        ),
        batched=True,
        num_proc=30,
        remove_columns=train_dataset.features.keys(),
        # load_from_cache_file=False,
        desc="tokenize",
    )
    dataset_list.append(train_dataset)
train_dataset = datasets.concatenate_datasets(dataset_list)

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

# import pdb
# pdb.set_trace()
# print(max(len(train_dataset['input_ids'])))
# train_dataset = dname2load[args.dataset](dataset_dir.get(args.dataset, None))
# logger.debug(train_dataset)

# 这个地方略有一点复杂，上面的train_dataset是原始的存储格式，在这一步，我们利用dname2func和template来把数据集转换成input_ids和labels
# 其中dname2func主要负责把原始数据集的格式重组成messages形式（即{'role':xxx , 'content':xxx}），template则负责把messages转成input_ids和labels
# train_dataset = train_dataset.map(
#     partial(dname2func[args.dataset], template=template),
#     batched=True,
#     num_proc=50,
#     remove_columns=train_dataset.features.keys(),
#     desc="tokenize",
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

    # from unsloth import FastLanguageModel

    # model = FastLanguageModel.get_peft_model(
    #     model,
    #     r = 8,
    #     target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
    #                     "gate_proj", "up_proj", "down_proj",
    #                     "lm_head", "embed_tokens",],
    #     lora_alpha = 16,
    # )

    model.print_trainable_parameters()


if args.output_dir is None:
    from datetime import datetime

    current_time = datetime.now()
    current_month = current_time.month
    current_day = current_time.day
    args.output_dir = f"naive_{args.model}_{args.dataset.replace(',','_')}_{current_month}m{current_day}d_bsz{args.total_bsz}_{args.lr_scheduler_type}_lr{args.learning_rate:.0e}"
    if args.lora:
        args.output_dir = args.output_dir + "_lora"
    if args.w_template:
        args.output_dir = args.output_dir + "_template"
    if args.label_smoothing_factor > 0:
        args.output_dir = args.output_dir + f"_ls{args.label_smoothing_factor}".replace(
            ".", ""
        )
    if args.warmup_steps > 0:
        args.output_dir = args.output_dir + f"_warmstep{args.warmup_steps}"
    if args.warmup_ratio > 0:
        args.output_dir = args.output_dir + f"_warmratio{args.warmup_ratio:.0e}"
    logger.info(f"未检测到output_dir，故采用自动生成的{args.output_dir}")


teacher_model_dir = model_dir.get(args.teacher_model, args.teacher_model)
from transformers import AwqConfig

quantization_config = AwqConfig()
teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_dir,
    low_cpu_mem_usage=True,
    # torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
).to("cuda:2")

trainer = KDTrainer(
    teacher_model=teacher_model,
    model=model,
    args=TrainingArguments(
        # optim="adamw_apex_fused",
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        # learning_rate=args.learning_rate,  # 学习率
        per_device_train_batch_size=real_bsz,  # 每个设备的训练批量大小
        num_train_epochs=args.num_train_epochs,  # 训练的轮次
        # weight_decay=args.weight_decay,
        # evaluation_strategy="epoch",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        dataloader_num_workers=8,
        bf16=True,
        logging_steps=1,
        remove_unused_columns=True,
        save_strategy="no",
        warmup_ratio=args.warmup_ratio,
        label_smoothing_factor=args.label_smoothing_factor,
    ),
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=my_collator,
)
# dataloader=trainer.get_train_dataloader()
# for d in dataloader:
#     input_ids=d['input_ids']

#     labels=d['labels']
#     mask = labels != -100
#     filtered_tensor = labels[mask]
#     print(tokenizer.convert_ids_to_tokens(input_ids[0]))
#     print(tokenizer.convert_ids_to_tokens(filtered_tensor))
#     # print(d)
#     import pdb
#     pdb.set_trace()
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

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
from niuload import balanced_load


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
    parser.add_argument("--template")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--lora", action="store_true", help="decide to use lora or not")
    parser.add_argument("--total_bsz", default=64, type=int)
    parser.add_argument("--label_smoothing_factor", default=0, type=float)
    parser.add_argument("--w_template", default=True, type=ast.literal_eval)
    parser.add_argument("--teacher_model", default=None)  # 添加teacher_model参数
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0)
    parser.add_argument("--lr_scheduler_type", default="linear")
    parser.add_argument("--learning_rate", default=5e-5, type=ast.literal_eval)
    return parser.parse_args()


def is_torchrun():
    args = sys.argv
    return "torchrun" in args or any(arg.startswith("--nproc_per_node") for arg in args)


def is_accelerate():
    args = sys.argv
    return "accelerate" in args


def main():
    args = parse_args()

    if is_torchrun():
        real_bsz = (
            args.total_bsz
            // args.gradient_accumulation_steps
            // torch.cuda.device_count()
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

    # 加载tokenizer
    real_model_dir = model_dir.get(args.model, args.model)
    tokenizer = AutoTokenizer.from_pretrained(real_model_dir)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # 加载模型
    if args.teacher_model:  # 知识蒸馏模式
        model = balanced_load(
            model_dir=real_model_dir,
            num_devices=4,
            ratio=[1, 1, 1, 1],
            devices_idx=[0, 1, 2, 3],
            is_distillation=True,
        )
    else:  # 普通训练模式
        model = AutoModelForCausalLM.from_pretrained(
            real_model_dir,
            torch_dtype="auto",
            device_map=(
                "balanced_low_0"
                if (not is_torchrun() and not is_accelerate())
                else None
            ),
            attn_implementation="eager" if "gemma2" in args.model else "sdpa",
        )

    # 准备模板和数据集
    template = modelType2Template[args.template](tokenizer)
    my_collator = MyCollator(tokenizer)

    # 处理数据集
    dataset_name_list = args.dataset.split(",")
    dataset_list = []
    for dname in dataset_name_list:
        train_dataset = dname2load[dname](dataset_dir.get(dname, None))
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
            load_from_cache_file=False,
            desc="tokenize",
        )
        dataset_list.append(train_dataset)
    train_dataset = datasets.concatenate_datasets(dataset_list)

    # LoRA配置
    if args.lora:
        from peft import LoraConfig, get_peft_model

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

    # 设置输出目录
    if args.output_dir is None:
        from datetime import datetime

        current_time = datetime.now()
        args.output_dir = f"{'kd' if args.teacher_model else 'naive'}_{args.model}_{args.dataset.replace(',','_')}_{current_time.month}m{current_time.day}d_bsz{args.total_bsz}_{args.lr_scheduler_type}_lr{args.learning_rate:.0e}"
        if args.lora:
            args.output_dir += "_lora"
        if args.w_template:
            args.output_dir += "_template"
        if args.label_smoothing_factor > 0:
            args.output_dir += f"_ls{args.label_smoothing_factor}".replace(".", "")
        if args.warmup_steps > 0:
            args.output_dir += f"_warmstep{args.warmup_steps}"
        if args.warmup_ratio > 0:
            args.output_dir += f"_warmratio{args.warmup_ratio:.0e}"

    # 训练配置
    training_args = TrainingArguments(
        seed=42,
        data_seed=42,
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=real_bsz,
        dataloader_prefetch_factor=2,
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        dataloader_num_workers=8,
        bf16=True,
        logging_steps=1,
        remove_unused_columns=True,
        save_strategy="epoch",
        warmup_ratio=args.warmup_ratio,
        label_smoothing_factor=args.label_smoothing_factor,
    )

    # 选择trainer
    if args.teacher_model:
        teacher_model = balanced_load(
            model_dir=model_dir.get(args.teacher_model, args.teacher_model),
            num_devices=4,
            ratio=[1, 1, 1, 1],
            devices_idx=[4, 5, 6, 7],
            is_distillation=False,
        )
        trainer = KDTrainer(
            teacher_model=teacher_model,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=my_collator,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            data_collator=my_collator,
        )

    # 训练和保存
    trainer.train()
    trainer.save_model(args.output_dir)
    trainer.save_state()

    # 保存参数
    saved_args_dict = vars(args)
    saved_args_dict["实际的总batch_size"] = (
        args.gradient_accumulation_steps * real_bsz * torch.cuda.device_count()
    )
    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as o:
        json.dump(saved_args_dict, o, ensure_ascii=False, indent=4)
    logger.info(f"模型被保存至{args.output_dir}")


if __name__ == "__main__":
    main()

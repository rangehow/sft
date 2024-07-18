from functools import partial
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


class MyCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, examples) -> torch.Any:
        input_ids = [list(example["input_ids"]) for example in examples]
        labels = [list(example["labels"]) for example in examples]

        input_ids_padded = self.tokenizer.pad(
            {"input_ids": input_ids}, return_tensors="pt", padding=True
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
    return parser.parse_args()


args = parse_args()

if is_torchrun:
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


model_dir = model_dir.get(args.model, args.model)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype="auto",
    device_map="auto",
    # attn_implementation="flash_attention_2" if args.fa2 else "sdpa",
    attn_implementation="sdpa",
)

# NOTE 从config.json中读取模型的类型，从而自动获取合适的模板类型
config = AutoConfig.from_pretrained(model_dir)
model_type = config.model_type
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
        # remove_columns=train_dataset.features.keys(),
        load_from_cache_file=False,
        desc="tokenize",
    )
    dataset_list.append(train_dataset)
train_dataset = datasets.concatenate_datasets(dataset_list)
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
    model.print_trainable_parameters()


trainer = Trainer(
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
        bf16=True,
        remove_unused_columns=True,
        save_strategy="epoch",
        label_smoothing_factor=args.label_smoothing_factor,
    ),
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=my_collator,
)
# dataloader=trainer.get_train_dataloader()
# for d in dataloader:
#     print(d)
#     import pdb
#     pdb.set_trace()
trainer.train()
trainer.save_model(args.output_dir)

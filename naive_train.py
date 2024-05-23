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
import torch


class MyCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, examples) -> torch.Any:
        input_ids = [list(example["input_ids"]) for example in examples]
        labels = [list(example["labels"]) for example in examples]

        input_ids_padded = self.tokenizer.pad(
            {"input_ids": input_ids}, return_tensors="pt", padding=True
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(label) for label in labels],
            batch_first=True,
            padding_value=-100,
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
    parser.add_argument("--output_dir")
    parser.add_argument("--learning_rate", default=5e-5)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--weight_decay", default=0.01)
    parser.add_argument("--gradient_accumulation_steps", default=4)
    # TODO 边写下面边思考，这里需要什么参数？
    return parser.parse_args()


args = parse_args()

model_dir = model_dir[args.model]
tokenizer = AutoTokenizer.from_pretrained(model_dir)
tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
tokenizer.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
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
train_dataset = datasets.load_dataset(dataset_dir[args.dataset])["train"]
# logger.debug(train_dataset)

# 这个地方略有一点复杂，上面的train_dataset是原始的存储格式，在这一步，我们利用dname2func和template来把数据集转换成input_ids和labels
# 其中dname2func主要负责把原始数据集的格式重组成messages形式（即{'role':xxx , 'content':xxx}），template则负责把messages转成input_ids和labels
train_dataset = train_dataset.map(
    partial(dname2func[args.dataset], template=template),
    batched=True,
    num_proc=50,
    remove_columns=train_dataset.features.keys(),
    desc="tokenize",
)

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        # optim="adamw_apex_fused",
        output_dir=args.output_dir,
        # learning_rate=args.learning_rate,  # 学习率
        per_device_train_batch_size=args.train_batch_size,  # 每个设备的训练批量大小
        num_train_epochs=args.num_train_epochs,  # 训练的轮次
        # weight_decay=args.weight_decay,
        # evaluation_strategy="epoch",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        bf16=True,
        remove_unused_columns=True,
        save_strategy='epoch',
    ),
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    data_collator=my_collator,
)
trainer.train()

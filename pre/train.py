# # 计算模型的总参数量
# total_params = sum(p.numel() for p in model.parameters())
# total_params_million = total_params / 1e6

# # 计算lm head的参数量
# lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
# lm_head_params_million = lm_head_params / 1e6

# # 打印结果
# print(f"Total number of parameters: {total_params_million:.2f}M")
# print(f"LM head parameters: {lm_head_params_million:.2f}M")

import torch.utils
from transformers import (
    LlamaConfig,
    AutoTokenizer,
    LlamaForCausalLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

config = LlamaConfig(
    hidden_size=1536,
    intermediate_size=4096,
    num_attention_heads=12,
    num_hidden_layers=12,
)
from torch.utils.data import DataLoader, Dataset
import os

os.environ["WANDB_PROJECT"] = "<my-amazing-project>"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints


tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B", padding_size="left"
)
tokenizer.pad_token = tokenizer.eos_token
model = LlamaForCausalLM(config)


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


dataset = CustomDataset(
    # 目标1----------------------
    [
        {
            "input_ids": [128, 129, 130],
            "labels": [-100, -100, 200],
        }
        for _ in range(2)
    ]
    + [
        {
            "input_ids": [128, 129, 130],
            "labels": [-100, -100, 201],
        }
        for _ in range(1)
    ]
    + [
        {
            "input_ids": [128, 129, 130],
            "labels": [-100, -100, 202],
        }
        for _ in range(1)
    ]
    # 干扰项1 局部相同前缀，且target里包含/不包含目标-------------------------------------------
    + [
        {
            "input_ids": [128, 129, 131],
            "labels": [-100, -100, 205],
        }
        for _ in range(1)
    ]
    + [
        {
            "input_ids": [128, 129, 132],
            "labels": [-100, -100, 202],
        }
        for _ in range(1)
    ]
    # 目标2
    + [
        {
            "input_ids": [1280, 1290, 1300],
            "labels": [-100, -100, 2000],
        }
        for _ in range(2)
    ]
    + [
        {
            "input_ids": [1280, 1290, 1300],
            "labels": [-100, -100, 2001],
        }
        for _ in range(1)
    ]
    + [
        {
            "input_ids": [1280, 1290, 1300],
            "labels": [-100, -100, 2002],
        }
        for _ in range(1)
    ]
    # 干扰项
    + [
        {
            "input_ids": [1280, 1290, 1301],
            "labels": [-100, -100, 2003],
        }
        for _ in range(1)
    ]
    + [
        {
            "input_ids": [1280, 1290, 1302],
            "labels": [-100, -100, 2002],
        }
        for _ in range(1)
    ]
    # 杂项---------------------------------------------------
    + [
        {
            "input_ids": [256, 257, 259],
            "labels": [-100, -100, i],
        }
        for i in range(5000, 5040)
    ]
)
import torch


def collate_fn(batch):

    input_ids = [b["input_ids"] for b in batch]
    labels = [b["labels"] for b in batch]
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels)}


evaluate_dataset = CustomDataset(
    [
        {
            "input_ids": [128, 129, 130],
            "labels": [-100, -100, 200],
        }
    ]
    + [
        {
            "input_ids": [1280, 1290, 1300],
            "labels": [-100, -100, 2000],
        }
    ]
)


# dataloader = DataLoader(dataset, collate_fn=collator, batch_size=2)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=100,
    num_train_epochs=10000,
    bf16=True,
    report_to="wandb",
    run_name="naive",
    save_strategy="no",
)

from torch.nn.functional import softmax


def compute_metrics(eval):

    pred = torch.nn.functional.softmax(
        torch.tensor(
            eval.predictions[
                :,
                -1,
            ][0]
        ),
        dim=-1,
    )
    target = torch.zeros_like(pred)
    target[200] = 1 / 2
    target[201] = 1 / 4
    target[202] = 1 / 4
    sim1 = torch.nn.functional.cosine_similarity(pred.unsqueeze(0), target.unsqueeze(0))

    pred = torch.nn.functional.softmax(
        torch.tensor(
            eval.predictions[
                :,
                -1,
            ][1]
        ),
        dim=-1,
    )
    target = torch.zeros_like(pred)
    target[2000] = 1 / 2
    target[2001] = 1 / 4
    target[2002] = 1 / 4
    sim2 = torch.nn.functional.cosine_similarity(pred.unsqueeze(0), target.unsqueeze(0))

    return {"sim1":sim1,"sim2":sim2,"similarity": (sim1 + sim2) / 2}


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    compute_metrics=compute_metrics,
    eval_dataset=evaluate_dataset,
    data_collator=collate_fn,
    tokenizer=tokenizer,
)


# Train the model
trainer.train()
# dataLoader = trainer.get_train_dataloader()
# for d in dataLoader:
#     import pdb

#     pdb.set_trace()

# # 计算模型的总参数量
# total_params = sum(p.numel() for p in model.parameters())
# total_params_million = total_params / 1e6

# # 计算lm head的参数量
# lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
# lm_head_params_million = lm_head_params / 1e6

# # 打印结果
# print(f"Total number of parameters: {total_params_million:.2f}M")
# print(f"LM head parameters: {lm_head_params_million:.2f}M")

from collections import Counter
import torch.utils
from transformers import (
    LlamaConfig,
    AutoTokenizer,
    LlamaForCausalLM,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
import ast
config = LlamaConfig(
    hidden_size=1536,
    intermediate_size=4096,
    num_attention_heads=12,
    num_hidden_layers=12,
)
from ..special_trainer import KLTrainer
from torch.utils.data import DataLoader, Dataset
import os

os.environ["WANDB_PROJECT"] = "<my-amazing-project>"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints

from argparse import ArgumentParser
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--div_mode", default=True, type=ast.literal_eval)
    return parser.parse_args()


args = parse_args()



tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B", padding_size="left"
)
tokenizer.pad_token = tokenizer.eos_token
model = LlamaForCausalLM(config)
embedding_size = model.lm_head.weight.size()[0]


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


from ..dataset import directly_softmax

dataset = CustomDataset(
    [
        # 目标1----------------------
        {
            "input_ids": [128, 129, 130],
            "labels": Counter({200: 2, 201: 1, 202: 1}),
        },
        {
            "input_ids": [128, 129, 131],
            "labels": Counter({205: 1}),
        },
        {
            "input_ids": [128, 129, 132],
            "labels": Counter({202: 1}),
        },
        # 目标2
        {
            "input_ids": [1280, 1290, 1300],
            "labels": Counter({2000: 2}),
        },
        {
            "input_ids": [1280, 1290, 1300],
            "labels": Counter({2001: 1}),
        },
        {
            "input_ids": [1280, 1290, 1300],
            "labels": Counter({2002: 1}),
        },
        {
            "input_ids": [1280, 1290, 1301],
            "labels": Counter({2003: 1}),
        },
        {
            "input_ids": [1280, 1290, 1302],
            "labels": Counter({2002: 1}),
        },
        {
            "input_ids": [256, 257, 259],
            "labels": Counter(list(range(5000, 5040))),
        },
    ]
)
import torch


def collate_fn(batch):

    input_ids = [b["input_ids"] for b in batch]

    labels = [b["labels"] for b in batch]
    labels = directly_softmax(labels, embedding_size,div=args.div_mode)
    valid_label_index_list =  [[(2, 3)] for _ in range(len(input_ids))]

    return {
        "input_ids": torch.tensor(input_ids),
        "all_prob_clm":labels,
        "attention_mask": None,
        "clm_cnt": None,
        "valid_label_index_list": valid_label_index_list,
    }


evaluate_dataset = CustomDataset(
    [
        {
            "input_ids": [128, 129, 130],
            "labels": Counter({2002: 1}),
        }
    ]
    + [
        {
            "input_ids": [1280, 1290, 1300],
            "labels":Counter({2002: 1}),
        }
    ]
    + [
        {
            "input_ids": [256, 257, 259],
            "labels":Counter({2002: 1}),
        }
    ]
)


# dataloader = DataLoader(dataset, collate_fn=collator, batch_size=2)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=32,
    num_train_epochs=10000,
    bf16=True,
    report_to="wandb",
    run_name="statistic_div" if args.div_mode else "statistic",
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
    
    
    target = torch.zeros_like(pred)
    target[5000:5040] = 1 / 40
    pred = torch.nn.functional.softmax(
        torch.tensor(
            eval.predictions[
                :,
                -1,
            ][2]
        ),
        dim=-1,
    )
    irrlevant_sim = torch.nn.functional.cosine_similarity(
        pred.unsqueeze(0), target.unsqueeze(0)
    )

    irrlevant_sim= torch.nn.functional.cosine_similarity(pred.unsqueeze(0), target.unsqueeze(0))

    return {"sim1": sim1, "sim2": sim2, "similarity": (sim1 + sim2) / 2,'irrlevant_sim':irrlevant_sim}


# Trainer
trainer = KLTrainer(
    pt_mode=True,
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

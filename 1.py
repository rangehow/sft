from niuload import balanced_load
import torch
from transformers import AutoTokenizer

from config import *

# 初始化tokenizer并准备输入
tokenizer = AutoTokenizer.from_pretrained("/mnt/rangehow/models/gemma-2-9b-it")
texts = [
    "Today is a beautiful day!",
    "How are you doing?"
]

# 创建批处理的输入
batch_encoding = tokenizer(
    texts,
    padding='max_length',
    max_length=128,
    truncation=True,
    return_tensors='pt'
)

input_ids = batch_encoding['input_ids']
attention_mask = batch_encoding['attention_mask']

# 加载模型并进行推理
teacher_model = balanced_load(
    model_dir="/mnt/rangehow/models/gemma-2-9b-it",
    num_devices=2,
    ratio=[0.5,1],
    devices_idx=[0,1],
)

with torch.inference_mode():
    teacher_outputs = teacher_model(
        input_ids=input_ids.to(teacher_model.device),
        attention_mask=attention_mask.to(teacher_model.device),
        use_cache=False
    )
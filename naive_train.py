from functools import partial
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
from config import *
from template import modelType2Template
from argparse import ArgumentParser
import datasets
from loguru import logger
from dataset_func import dname2func



def parse_args():
    parser=ArgumentParser()
    # TODO 思考一下，这里需要什么参数？
    
    
    return parser.parse_args()


args = parse_args()


tokenizer = AutoTokenizer.from_pretrained(model_dir[args.model])

# NOTE 从config.json中读取模型的类型，从而自动获取合适的模板类型
config=AutoConfig.from_pretrained(model_dir[args.model])
model_type=config.model_type
template=modelType2Template[model_type](tokenizer)

# 读取数据集
train_dataset = datasets.load_dataset(
    dataset_dir[args.dataset]
)["train"]
logger.debug(train_dataset)

# 这个地方略有一点复杂，上面的数据集是他原始的存储格式，在这一步，我们利用dname2func和template来把数据集转换成input_ids和labels
# 其中dname2func主要负责把原始数据集的格式重组成messages形式（即{'role':xxx , 'content':xxx}）
train_dataset = train_dataset.map(
    partial(dname2func[args.dataset], template=template),
    batched=True,
    num_proc=50,
    remove_columns=train_dataset.features.keys(),
    desc="tokenize",
)
# TODO 先考虑hf现成的collator能不能使用，实现一个collator

# TODO 直接初始化一个trainer




trainer.train()

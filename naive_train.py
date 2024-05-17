from functools import partial
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig,Trainer,DataCollator
from config import *
from template import modelType2Template
from argparse import ArgumentParser
import datasets
from loguru import logger
from dataset_func import dname2func

class MyCollator(DataCollator):
    def __call__(self, examples):
        # examples 是一个包含了所有样本的列表
        input_ids = [example['input_ids'] for example in examples]  
        labels = [example['labels'] for example in examples]  

        return {
            'input_ids': input_ids,  # 返回模型输入的文本数据
            'labels': labels  # 返回对应的标签数据
        }

def parse_args():
    parser=ArgumentParser()
    parser.add_argument("--model", default="/data/ruanjh/best_training_method/gemma-2b")
    parser.add_argument("--dataset", default="/data_path")
    parser.add_argument("--output_dir", default="output_path")
    parser.add_argument("--learning_rate", default=5e-5)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--weight_decay", default=0.01)
    # TODO 边写下面边思考，这里需要什么参数？
    return parser.parse_args()

args = parse_args()

tokenizer = AutoTokenizer.from_pretrained(model_dir[args.model])

# NOTE 从config.json中读取模型的类型，从而自动获取合适的模板类型
config=AutoConfig.from_pretrained(model_dir[args.model])
model_type=config.model_type
template=modelType2Template[model_type](tokenizer)
my_collator= MyCollator()
# 读取数据集
train_dataset = datasets.load_dataset(
    dataset_dir[args.dataset]
)["train"]
logger.debug(train_dataset)

# 这个地方略有一点复杂，上面的train_dataset是原始的存储格式，在这一步，我们利用dname2func和template来把数据集转换成input_ids和labels
# 其中dname2func主要负责把原始数据集的格式重组成messages形式（即{'role':xxx , 'content':xxx}），template则负责把messages转成input_ids和labels
train_dataset = train_dataset.map(
    partial(dname2func[args.dataset], template=template),
    batched=True,
    num_proc=50,
    remove_columns=train_dataset.features.keys(),
    desc="tokenize",
)

# TODO 先考虑hf现成的collator能不能使用，实现一个collator

# TODO 直接初始化一个trainer
trainer = Trainer(
    model=args.model,
    config= config,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    output_dir=args.output_dir,          # 输出目录
    evaluation_strategy="epoch",     # 每个epoch结束时进行评估
    learning_rate=args.learning_rate,              # 学习率
    per_device_train_batch_size=args.train_batch_size,  # 每个设备的训练批量大小
    num_train_epochs=args.num_train_epochs,       # 训练的轮次
    weight_decay=args.weight_decay, 
    data_collator=my_collator,              # 权重衰减，防止过拟合
    bf16 = True,
    remove_unused_columns=True,
)

if __name__== "__main__":
    trainer.train()
from functools import partial
from transformers import AutoTokenizer,AutoModelForCausalLM,AutoConfig
from config import *
from template import modelType2Template
from argparse import ArgumentParser
import datasets
from loguru import logger
from dataset_func import dname2func
import torch
from chat_template import modelType2ChatTemplate

def parse_args():
    parser=ArgumentParser()
    parser.add_argument('--mode',type=int,help='0: base model(wo template),1:instruct model')
    parser.add_argument('--model')
    
    
    return parser.parse_args()
args=parse_args()
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda()

text="Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
if args.mode==1:
    input_text = [{'role':'user','content':text}]
    
    if tokenizer.chat_template is None:
        tokenizer.chat_template = modelType2ChatTemplate[model.config.model_type]
    input_ids = tokenizer.apply_chat_template(input_text,add_generation_prompt=True,return_tensors='pt').to('cuda')
    
    print(input_ids)

    outputs = model.generate(input_ids,max_new_tokens=1000)
else:
    input_ids = tokenizer(text,return_tensors='pt').to('cuda')
    print(input_ids)
    outputs = model.generate(**input_ids,max_new_tokens=1000)


print(tokenizer.decode(outputs[0]))

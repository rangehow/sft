"""
https://github.com/google-deepmind/gemma/blob/main/colabs/gsm8k_eval.ipynb
"""

import ast
from functools import partial
import json
import pickle
import datasets
import re
from loguru import logger
import os
from argparse import ArgumentParser
import torch
from torch.nn.parallel import DataParallel
from tqdm import tqdm
from ..dataset_func import dname2func
from ..template import modelType2Template
from transformers import AutoTokenizer, AutoConfig, DataCollatorForSeq2Seq,AutoModelForCausalLM
import os
from ..eval.post_process import dname2post
from torch.utils.data import DataLoader
from ..eval.load_func import dname2load
from ..eval.samplingparam import dname2samplingparams
from vllm import LLM
from ..config import *
from ..dataset import SpecialDataset,SpecialDataCollator
import torch.nn.functional as F

def parse_args():
    parser = ArgumentParser()

    parser.add_argument(
        "--reuse",
        action="store_true",
        help="在批量测试时自动开reuse",
    )
    parser.add_argument(
        "--dataset",
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--output_path",
    )
    parser.add_argument("--zero_prob", default=0, type=ast.literal_eval)
    parser.add_argument("--div_mode", default=False, type=ast.literal_eval)
    parser.add_argument(
        "--mix",default=False, type=ast.literal_eval, help="decide to use token level freq weight"
    )
    parser.add_argument(
        "--mix_ratio",default=0.8, type=ast.literal_eval, help="sft信号的融合比例"
    )
    return parser.parse_args()



@logger.catch
def main():
    args = parse_args()

    model_list=args.model.split(',')
    dataset_list=args.dataset.split(',')

    # os.makedirs(args.output_path,exist_ok=True)
    script_path = os.path.dirname(os.path.abspath(__file__))
    print("script_path", script_path)
    for m in model_list:
        model_type = AutoConfig.from_pretrained(
                os.path.join(m, "config.json")
            ).model_type
        tokenizer = AutoTokenizer.from_pretrained(m)
        tokenizer.padding_side = "left"
        template = modelType2Template[model_type](tokenizer)
        model_name = os.path.basename(
                m.rstrip(os.sep)
            )  # 不去掉sep，碰到 a/b/ 就会读到空。
        record_list=[]
        
        for d in dataset_list:
            
            
            
            save_str = (
                f"{model_name}_{d}_{args.div_mode}_{args.zero_prob}"
            )
            print("save_str", save_str)
            target_file = os.path.join(script_path, "generated", save_str)
            print("target_file", target_file)
            reuse_flag = True if args.reuse and os.path.exists(target_file) else False
            if not args.reuse:
                if os.path.exists(target_file):
                    while True:
                        i = input("本次任务似乎已经被完成过了~输入y可以复用，输入n则重新生成：")
                        if i == "y":
                            reuse_flag = True
                            break
                        elif i == "n":
                            break
                        else:
                            print("输入错误，必须是y或n")

            if reuse_flag:
                with open(target_file, "rb") as r:
                    response = pickle.load(r)
            else:
                @logger.catch
                def load_dataset():

                    data_folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__).rstrip(os.sep)).rstrip(os.sep))
                    with open(
                        f"{data_folder_path}/train_dataset/{model_type}_{args.dataset}_synthesis.pkl", "rb"
                    ) as f:
                        synthesis = pickle.load(f)

                    with open(
                        f"{data_folder_path}/train_dataset/{model_type}_{args.dataset}_index.pkl", "rb"
                    ) as f:
                        index = pickle.load(f)

                    train_dataset = SpecialDataset(
                        synthesis,
                        index,
                        embedding_size=-1, # 遗留行为，dataset用不上这个参数
                        zero_prob=args.zero_prob,
                        div_mode=args.div_mode,
                    )
                    return train_dataset
                train_dataset = load_dataset()
                
                with torch.inference_mode():
                    model = AutoModelForCausalLM.from_pretrained(m,attn_implementation="flash_attention_2",torch_dtype="auto",).cuda()
                    tokenizer=AutoTokenizer.from_pretrained(m,padding_side='left')
                    collator=SpecialDataCollator(tokenizer=tokenizer,zero_prob=args.zero_prob,embedding_size = model.lm_head.weight.size()[0],div_mode=args.div_mode)
                    dataloader=DataLoader(dataset=train_dataset,batch_size=8,collate_fn=collator,num_workers=16,pin_memory=True )
                    supervised_similarity,clm_similarity,naive_label_similarity=0,0,0
                    for d in tqdm(dataloader):
                        response = model(input_ids=d['input_ids'].to('cuda'),attention_mask=d['attention_mask'].to('cuda')).logits
                        last_logits = torch.cat(
                            [
                                row[start:end]
                                for row, turn in zip(response, d['valid_label_index_list'])
                                for start, end in turn
                            ]
                        ).to('cpu')
                        real_label = torch.cat(
                            [
                                torch.cat((row[start+1:end],torch.tensor([tokenizer.eos_token_id])))
                                for row, turn in zip(d['input_ids'], d['valid_label_index_list'])
                                for start, end in turn
                            ]
                        )
                        label_tensor = F.one_hot(real_label, model.lm_head.weight.size()[0])

                        def is_one_hot(vector):
                            # 检查是否只有一个1，其余为0
                            return torch.sum(vector) == 1 and torch.all(torch.logical_or(vector == 0, vector == 1))
                        def find_non_one_hot_rows(tensor):
                            non_one_hot_indices = []
                            for i, row in enumerate(tensor):
                                if not is_one_hot(row):
                                    non_one_hot_indices.append(i)
                            return non_one_hot_indices
                        
                        all_prob_supervised = d['all_prob_supervised'].to('cpu')
                        all_prob_clm = d['all_prob_clm'].to('cpu')
                        temp_supervised_similarity,temp_clm_similarity=F.cosine_similarity(last_logits, all_prob_supervised),F.cosine_similarity(last_logits, all_prob_clm)
                        temp_label_similarity=F.cosine_similarity(last_logits, label_tensor)
                        supervised_similarity+=torch.sum(temp_supervised_similarity).item()
                        clm_similarity+=torch.sum(temp_clm_similarity).item()
                        naive_label_similarity+=torch.sum(temp_label_similarity).item()
                        
                        
                        
                        not_equal = torch.ne(all_prob_supervised, label_tensor)
                        diff_indices = torch.nonzero(not_equal, as_tuple=True)[0]
                        if torch.numel(diff_indices)!=0:
                            import pdb
                            pdb.set_trace()
                        aaaa=find_non_one_hot_rows(all_prob_supervised)
                        bbbb=find_non_one_hot_rows(all_prob_clm)
                        
                    print(supervised_similarity,clm_similarity,naive_label_similarity)

                




            
        


if __name__ == "__main__":
    main()

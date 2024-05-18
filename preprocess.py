from functools import partial
import gc
import os
import time
import torch
from collections import Counter, defaultdict

# from dataset import CapybaraDataset
from transformers import AutoTokenizer,AutoConfig
import json
import pickle
import pdb
import warnings
import datasets
import ast
from config import model_dir, dataset_dir
from dataset_func import dname2func
from template import modelType2Template

from tqdm import tqdm
from loguru import logger

warnings.filterwarnings("ignore", "The iteration is not making good progress")


import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma_2b")
    parser.add_argument("--dataset", default="alpaca_cleaned")
    parser.add_argument("--clm", default=True, type=ast.literal_eval)
    parser.add_argument("--ngram", default=4)
    parser.add_argument('--cache_statistic',default=True, type=ast.literal_eval)
    return parser.parse_args()


def find_ranges(lst, target=-100):
    ranges = []
    start = None
    multiTurnOnlyOnceInfoFlag=True
    for i, num in enumerate(lst):
        if num != target and start is None:
            start = i
        elif num == target and start is not None:
            if multiTurnOnlyOnceInfoFlag:
                logger.info('这个分支理论上只有多轮对话的数据集才会进入,确保自己在使用多轮对话数据集')
                multiTurnOnlyOnceInfoFlag=False
            #  -100（start-1） start ，，，words4predictend(i-2) end(i-1) -100（i） 这个数据结构被用于从model_output里按切片取出logits来预测下一个词
            ranges.append((start-1, i-1))  # 因为是切片，i-1实际上会取到i-2范围,logits的核心就是不要预测任何-100
            start = None
    
    if start is not None:
        # 这个地方结束位置一般不重要，除非最后有什么不需要预测的特殊标志。
        ranges.append((start-1, len(lst)-1))
    
    return ranges


@logger.catch
def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(model_dir[args.model])
    config=AutoConfig.from_pretrained(model_dir[args.model])
    model_type=config.model_type
    template=modelType2Template[model_type](tokenizer)

    train_dataset = datasets.load_dataset(
        dataset_dir[args.dataset]
    )["train"]
    print(train_dataset)
    train_dataset = train_dataset.map(
        partial(dname2func[args.dataset], template=template),
        batched=True,
        num_proc=30,
        remove_columns=train_dataset.features.keys(),
        desc="tokenize",
    )
    import pdb
    pdb.set_trace()
    def statistic():
        
        supervised_dict = defaultdict(Counter)
        if args.clm:
            clm_dict = defaultdict(Counter)


        logger.debug(f"start to make statistic")
        # 统计begin-----------------------------------------------------------------------------------
        for j in tqdm(range(len(train_dataset)), desc="statistic stage"):
            
            input_id, label = train_dataset[j]["input_ids"], train_dataset[j]["labels"]

            # 如果是多轮对话,那么label将是穿插着多段连续-100的int list
            # supervised信号的key是第一段-100结束开始,以后开始递增
            # clm信号的key应该是非-100区域内独立统计
            if args.clm:
                # 用于标志是否到达非-100区域的,这里有个假定就是开头一定是连续的-100区域[通常因为开头是特殊标记,所以总是的]
                # 这个标记主要的作用就是为了辅助regionBeginIdx更新
                flag4LossArea = False
                # 用于辅助ngram测算现在是否可以更新clm了
                regionBeginIdx = -1

            for i in range(len(label)-1):
                
                if label[i+1] != -100:
                    
                    supervised_key = tuple(input_id[:i+1])
                    supervised_dict[supervised_key].update([label[i+1]])

                    if args.clm:
                        if flag4LossArea is False:
                            # 此时下一个label不是-100，但是regionBeginIdx本身指向的还是-100
                            regionBeginIdx = i
                            flag4LossArea = True

                        if i - regionBeginIdx >= args.ngram:
                            clm_key = tuple(label[regionBeginIdx+1:i+1])
                            clm_dict[clm_key].update([label[i+1]])

                elif args.clm and flag4LossArea:
                    flag4LossArea = False

        logger.debug(f"supervised_dict,{len(supervised_dict)}")
        if args.clm:
            logger.debug(f"clm_dict,{len(clm_dict)}")

        return supervised_dict, clm_dict
        

    supervised_dict, clm_dict = statistic()

    def synthesis():    
        synthesis_dict = defaultdict(list)
        cnt_list=[]

        for j in tqdm(range(len(train_dataset)), desc="synthesis stage"):
            
            input_id, label = train_dataset[j]["input_ids"], train_dataset[j]["labels"]
            
            
           
            if args.clm:
                # 用于标志是否到达非-100区域的,这里有个假定就是开头一定是连续的-100区域[通常因为开头是特殊标记,所以总是的]
                # 这个标记主要的作用就是为了辅助regionBeginIdx更新
                flag4LossArea = False
                # 用于辅助ngram测算现在是否可以更新clm了
                regionBeginIdx = -1
            
            # 这个地方和encoder-decoder模型还不一样，不需要特地区分编解码的输入，所以只需要一个input_id即可，input_id最后的EOS不需要送给模型
            key = (tuple(input_id[:-1]))
            length = len(input_id)
            if synthesis_dict[key]==[] or  tokenizer.eos_token_id not in synthesis_dict[key][-1][0]: #防止重复示例:
                # cnt list必须在这里，不然对synthesis_dict的去重会导致长度不匹配
                cnt_list.append(find_ranges(label))
                
                for i in range(length-1): # 这个地方保证了 比如 -100 // non_-100_start_area ，，，words_4_predict_end(i-1) // end(i)
                    
                    if label[i+1] != -100:  #  // -100（start-1） non_-100_start_area ，，，words_4_predict_end(i-1) // end(i) -100（i+1） 实际上只统计 //内的区域
                        
                            
                            supervised_key = tuple(input_id[:i+1])
                            supervised_value=supervised_dict[supervised_key]

                            if args.clm:
                                if flag4LossArea is False:
                                    # 此时下一个label不是-100，但是regionBeginIdx本身指向的还是-100
                                    regionBeginIdx = i
                                    flag4LossArea = True

                                clm_key = tuple(label[regionBeginIdx+1:i+1])
                                if clm_key in clm_dict:
                                    clm_value=clm_dict[clm_key]
                                else:
                                    clm_value=supervised_value
                                    
                            synthesis_dict[key].append([supervised_value,clm_value])

                    elif args.clm and flag4LossArea:
                        flag4LossArea = False
            
        return synthesis_dict,cnt_list
            
        
    synthesis_dict,cnt_list=synthesis()
    assert len(synthesis_dict)==len(cnt_list)
    logger.debug(f'len(synthesis_dict)={len(synthesis_dict)},len(cnt_list)={len(cnt_list)}')
    with open(f"{dataset_dir[args.dataset]}/synthesis.pkl",'wb',) as o:
        pickle.dump(synthesis_dict, o,protocol=5)

    with open(f"{dataset_dir[args.dataset]}/index.pkl",'wb',) as o:
        pickle.dump(cnt_list,  o,protocol=5)
    logger.debug(f'整合文件被保存到{dataset_dir[args.dataset]}')


   
def test():
    args = parse_args()
    
    with open(f"{dataset_dir[args.dataset]}/index.pkl",'rb',) as o:
        cnt_list=pickle.load(o)
    with open(f"{dataset_dir[args.dataset]}/synthesis.pkl",'rb',) as o:
        synthesis_dict=pickle.load(o)
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir[args.model])
    config=AutoConfig.from_pretrained(model_dir[args.model])
    model_type=config.model_type
    template=modelType2Template[model_type](tokenizer)

    train_dataset = datasets.load_dataset(
        dataset_dir[args.dataset]
    )["train"]

    train_dataset = train_dataset.map(
        partial(dname2func[args.dataset], template=template),
        batched=True,
        num_proc=30,
        remove_columns=train_dataset.features.keys(),
        desc="tokenize",
    )
    
    synthesis_dict = [data_sample for data_sample in synthesis_dict.items()]
    cnt=0
    for i in range(len(synthesis_dict)):

        input_ids=synthesis_dict[i][0]
        length=cnt_list[i][0][-1]
        if len(input_ids)!=length:
            pdb.set_trace()
        # logger.debug(train_dataset[cnt])
        
        # cnt+=1
    logger.debug(len(synthesis_dict))
    
if __name__ == "__main__":
    # test()
    main()

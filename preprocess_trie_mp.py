from functools import partial
import os
from collections import Counter, defaultdict, deque
from transformers import AutoTokenizer, AutoConfig
from itertools import islice
import pickle
import warnings
import datasets
import ast
from config import model_dir, dataset_dir
from dataset_func import dname2func
from template import modelType2Template
from eval.load_func import dname2load
from tqdm import tqdm
from loguru import logger
import argparse
import multiprocessing
from datasets.distributed import split_dataset_by_node
warnings.filterwarnings("ignore", "The iteration is not making good progress")

import sys
# sys.setrecursionlimit(10000)  # Increase as needed
class TrieNode:
    def __init__(self):
        self.children = {}
        self.value = Counter()
    
    def __getstate__(self):
        return (self.children, self.value)

    def __setstate__(self, state):
        self.children, self.value = state

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def __getstate__(self):
        serialized = []
        queue = deque([([], self.root)])
        
        while queue:
            path, node = queue.popleft()
            serialized.append((path, dict(node.value)))
            for char, child in node.children.items():
                queue.append((path + [char], child))
        
        return serialized

    def __setstate__(self, state):
        self.root = TrieNode()
        for path, value in state:
            node = self.root
            for char in path:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
            node.value.update(value)
        
        
    def insert(self, key_list, value):
        node = self.root
        for key in key_list:
            if key not in node.children:
                node.children[key] = TrieNode()
            node = node.children[key]
        node.value[value] += 1

    def search(self, key_list):
        node = self.root
        for key in key_list:
            if key not in node.children:
                return None
            node = node.children[key]
        return node.value

    def merge(self, other):
        self._merge_nodes(self.root, other.root)

    def _merge_nodes(self, node1, node2):
        # Merge the values
        node1.value.update(node2.value)
        # Merge the children
        for key, child_node2 in node2.children.items():
            if key in node1.children:
                self._merge_nodes(node1.children[key], child_node2)
            else:
                node1.children[key] = child_node2
        
    # def __str__(self):
    #     def _print(node, prefix):
    #         if node.value:
    #             print(f"{prefix}: {dict(node.value)}")
    #         for key, child in node.children.items():
    #             _print(child, prefix + [key])
    #     _print(self.root, [])

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma_2b")
    parser.add_argument(
        "--dataset",
    )
    parser.add_argument("--clm", default=True, type=ast.literal_eval)
    parser.add_argument("--ngram", default=0, type=int)
    parser.add_argument("--cache_statistic", default=True, type=ast.literal_eval)
    parser.add_argument("--template", type=str)
    parser.add_argument("--mono", default=False, type=ast.literal_eval)
    parser.add_argument("--mono_dataset", default="wiki_medical")
    parser.add_argument("--w_template", default=True, type=ast.literal_eval)
    return parser.parse_args()

def find_ranges(lst, target=-100):
    ranges = []
    start = None
    multiTurnOnlyOnceInfoFlag = True
    have_target = False
    for i, num in enumerate(lst):
        if num==target:
            have_target=True
        
        if num != target and start is None:
            start = i
        elif num == target and start is not None:
            
            if multiTurnOnlyOnceInfoFlag:
                logger.info(
                    "这个分支理论上只有多轮对话的数据集才会进入,确保自己在使用多轮对话数据集"
                )
                multiTurnOnlyOnceInfoFlag = False
            #  -100（start-1） start ，，，words4predictend(i-2) end(i-1) -100（i） 这个数据结构被用于从model_output里按切片取出logits来预测下一个词

            ranges.append(
                (start - 1, i - 1)
            )  # 因为是切片，i-1实际上会取到i-2范围,logits的核心就是不要预测任何-100
            start = None

    if start is not None:
        # 这个地方结束位置一般不重要，除非最后有什么不需要预测的特殊标志。
        # 到底什么情况会进这里呢？整句话都不存在-100的时候
        if have_target:
            ranges.append((start - 1, len(lst) - 1))
        else:
            ranges.append((0, len(lst) - 1))

    return ranges

def chunk_data(data, chunk_size):
    """将大字典或大列表拆分成多个较小的部分，每个包含不超过 chunk_size 个元素。"""
    if isinstance(data, dict):
        it = iter(data)
        for _ in range(0, len(data), chunk_size):
            yield {k: data[k] for k in islice(it, chunk_size)}
    elif isinstance(data, list):
        for i in range(0, len(data), chunk_size):
            yield data[i : i + chunk_size]
    else:
        raise TypeError("data must be a dictionary or a list")

def save_chunks(data, chunk_size, base_dir, name, start_idx=0):
    """将大字典分块并保存到多个文件中。"""
    for i, chunk in tqdm(enumerate(chunk_data(data, chunk_size)), desc="Saving chunks", ascii=True):
        filename = f"{name}_part{i+start_idx}.msgpack"
        with open(os.path.join(base_dir, filename), "wb") as f:
            pickle.dump(chunk, f, protocol=5)
        print(f"Saved chunk {i} to {filename}")

def parse_dataset(args, template, dataset_str):
    dataset_name_list = dataset_str.split(",")
    dataset_list = []
    for dname in dataset_name_list:
        train_dataset = dname2load[dname](dataset_dir.get(dname, None))
        # train_dataset = datasets.load_dataset(dataset_dir.get(args.dataset, args.dataset))[
        #     "train"
        # ]

        print("\n数据集", dname, "=")
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
            remove_columns=train_dataset.features.keys(),
            load_from_cache_file=False,
            desc="tokenize",
        )
        dataset_list.append(train_dataset)
    return datasets.concatenate_datasets(dataset_list)

def process_chunk_statistic(chunk_data, args_ngram, args_clm):
    local_supervised_trie = Trie()
    local_clm_trie = Trie() if args_clm else None
    local_max_target_len = 0

    for data_point in chunk_data:
        
        input_id, label = data_point["input_ids"], data_point["labels"]

        assert len(input_id) == len(label)
        if args_clm:
            flag4LossArea = False
            regionBeginIdx = -1

        local_max_target_len = max(local_max_target_len, len(label))
        for i in range(len(label) - 1):
            if label[i + 1] != -100:
                local_supervised_trie.insert(input_id[: i + 1], label[i + 1])

                if args_clm:
                    if flag4LossArea is False:
                        regionBeginIdx = i
                        flag4LossArea = True

                    if i - regionBeginIdx >= args_ngram:
                        clm_key = label[regionBeginIdx + 1 : i + 1]
                        local_clm_trie.insert(clm_key, label[i + 1])
            elif args_clm and flag4LossArea:
                flag4LossArea = False

    return local_supervised_trie, local_clm_trie, local_max_target_len

def process_mono_chunk(chunk_data, args_ngram, max_target_len):
    local_clm_trie = Trie()
    for data_point in chunk_data:
        input_id, label = data_point["input_ids"], data_point["labels"]
        index_for_start_area = 0
        for i in range(len(label) - 1):
            if label[i] == -100:
                index_for_start_area += 1
            if (
                label[i + 1] != -100
                and i - index_for_start_area >= args_ngram - 1
                and i - index_for_start_area <= max_target_len + 5
            ):
                clm_key = label[index_for_start_area : i + 1]
                local_clm_trie.insert(clm_key, label[i + 1])
    return local_clm_trie

# Define the modified statistic function (with multiprocessing)
def statistic(args, train_dataset, mono_dataset=None):
    import math


    num_workers = multiprocessing.cpu_count()
    num_workers = max(1, min(num_workers - 1, len(train_dataset)))

    supervised_trie = Trie()
    clm_trie = Trie() if args.clm else None
    max_target_len = 0
    
    

    chunks = [split_dataset_by_node(train_dataset, rank=j, world_size=num_workers*100) for j in range(num_workers)]
    pool_args = [(chunk, args.ngram, args.clm) for chunk in chunks]

    with multiprocessing.Pool(num_workers) as pool:
        batch_results = pool.starmap(process_chunk_statistic, pool_args)

    for local_supervised_trie, local_clm_trie, local_max_target_len in batch_results:
        supervised_trie.merge(local_supervised_trie)
        if args.clm and local_clm_trie:
            clm_trie.merge(local_clm_trie)
        max_target_len = max(max_target_len, local_max_target_len)

    if args.mono and args.clm:
        chunk_size = math.ceil(len(mono_dataset) / num_workers)
        mono_chunks = [
            mono_dataset[i:i+chunk_size]
            for i in range(0, len(mono_dataset), chunk_size)
        ]

        pool_args = [(chunk, args.ngram, max_target_len) for chunk in mono_chunks]

        with multiprocessing.Pool(num_workers) as pool:
            mono_results = list(tqdm(pool.starmap(process_mono_chunk, pool_args), desc="Processing mono chunks", total=len(mono_chunks), ascii=True))

        for local_clm_trie in mono_results:
            clm_trie.merge(local_clm_trie)
    else:
        clm_trie = clm_trie if args.clm else None

    return supervised_trie, clm_trie

def synthesis(args, train_dataset, supervised_trie, clm_trie, template):
    tokenizer = template.tokenizer
    synthesis_dict = defaultdict(list)
    cnt_list = []
    logger.debug(f"start to synthesis")
    for j in tqdm(range(len(train_dataset)), desc="synthesis stage", ascii=True):
        
        input_id, label = train_dataset[j]["input_ids"], train_dataset[j]["labels"]

        if args.clm:
            # 用于标志是否到达非-100区域的,这里有个假定就是开头一定是连续的-100区域[通常因为开头是特殊标记,所以总是的]
            # 这个标记主要的作用就是为了辅助regionBeginIdx更新
            flag4LossArea = False
            # 用于辅助ngram测算现在是否可以更新clm了
            regionBeginIdx = -1

        # 这个地方和encoder-decoder模型还不一样，不需要特地区分编解码的输入，所以只需要一个input_id即可，input_id最后的EOS不需要送给模型
        key = tuple(input_id[:-1])
        length = len(input_id)
        if synthesis_dict[key] == [] or (
            template.chat_eos_token_id not in synthesis_dict[key][-1][0]
            and template.chat_eos_token_id
            not in synthesis_dict[key][-2][
                0
            ]  # qwen的template结尾是\n，我无语了。。
        ):  # 防止重复示例. 情况1，这条数据没有被添加过了，情况2，这条数据没有被添加到结束符
            # cnt list必须在这里，不然对synthesis_dict的去重会导致长度不匹配
            cnt_list.append(find_ranges(label))

            for i in range(
                length - 1
            ):  # 这个地方保证了 比如 -100 // non_-100_start_area ，，，words_4_predict_end(i-1) // end(i)
                
                if (
                    label[i + 1] != -100
                ):  #  // -100（start-1） non_-100_start_area ，，，words_4_predict_end(i-1) // end(i) -100（i+1） 实际上只统计 //内的区域
                    
                    # supervised_key = tuple(input_id[: i + 1])
                    # supervised_value = supervised_dict[supervised_key]
                    supervised_value = supervised_trie.search(input_id[: i + 1])

                    if args.clm:
                        if flag4LossArea is False:
                            # 此时下一个label不是-100，但是regionBeginIdx本身指向的还是-100
                            regionBeginIdx = i
                            flag4LossArea = True

                        # clm_key = tuple(label[regionBeginIdx + 1 : i + 1])
                        # clm_value = clm_dict.get(clm_key, supervised_value)
                        clm_value = clm_trie.search(
                            label[regionBeginIdx + 1 : i + 1]
                        )

                        if (
                            clm_value == None or len(clm_value) == 0
                        ):  # trie的返回不稳定，现在是空counter
                                
                            clm_value = supervised_value
                    try:
                        assert clm_value is not None
                        synthesis_dict[key].append([supervised_value, clm_value])
                    except:
                        import pdb
                        pdb.set_trace()

                elif args.clm and flag4LossArea:
                    flag4LossArea = False
                
                if len(synthesis_dict) != len(cnt_list):
                    # llama3-base和it的eos token不一样
                    import pdb

                    pdb.set_trace()

    return synthesis_dict, cnt_list



@logger.catch
def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(model_dir.get(args.model, args.model))
    # config = AutoConfig.from_pretrained(model_dir[args.model])
    # model_type = config.model_type
    # template = modelType2Template[model_type](tokenizer)
    template = modelType2Template[args.template](tokenizer)
    
    train_dataset = parse_dataset(args, template, args.dataset)
    # train_dataset = train_dataset.sort('input_ids')
    if args.mono:
        mono_dataset = parse_dataset(args, template, args.mono_dataset)
        supervised_trie, clm_trie = statistic(args,train_dataset,mono_dataset)
    else:
        supervised_trie, clm_trie = statistic(args,train_dataset)
    synthesis_dict, cnt_list = synthesis(args,train_dataset,supervised_trie, clm_trie,template)

    logger.debug(
        f"length of synthesis_dict:{len(synthesis_dict)};length of cnt_list:{len(cnt_list)}"
    )
    assert len(synthesis_dict) == len(cnt_list)

    script_path = os.path.dirname(os.path.abspath(__file__).rstrip(os.sep))
    # w_template
    output_base_dir = os.path.join(
        script_path,
        "train_dataset",
        (f"{args.template}_{args.dataset}"),
    )
    if args.mono:
        output_base_dir += f"_mono_{args.mono_dataset.replace(',','_')}"
    if args.w_template:
        output_base_dir += "_template"

    os.makedirs(
        output_base_dir,
        exist_ok=True,
    )

    save_chunks(
        synthesis_dict,
        chunk_size=1000,
        base_dir=output_base_dir,
        name="synthesis",
    )
    save_chunks(
        cnt_list,
        chunk_size=1000,
        base_dir=output_base_dir,
        name="index",
    )

    logger.debug(
        f"整合文件被保存到train_dataset/{args.template}_{args.dataset}"
        if not args.mono
        else f"{args.template}_{args.dataset}_mono_{args.mono_dataset.replace(',','_')}"
    )
    
    
    

if __name__ == "__main__":
    main()
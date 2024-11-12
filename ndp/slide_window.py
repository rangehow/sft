from functools import partial
import json
import math
import os
from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoConfig
from itertools import islice
import pickle
import warnings
import datasets
import ast
from ..config import model_dir, dataset_dir
from ..dataset_func import dname2func
from ..template import modelType2Template
from ..eval.load_func import dname2load
from tqdm import tqdm
from loguru import logger
import argparse
from ipdb import set_trace as bp


def find_ranges(lst, target=-100, offset=0):
    # explanation about offset：项目里有两个使用这个函数的地方，一个是找到label里的非0区域，另一个是写入最后数据里用来指示要提取出logits什么区域的，这俩offset差1
    ranges = []
    start = None
    multiTurnOnlyOnceInfoFlag = True
    have_target = False
    for i, num in enumerate(lst):
        if num == target:
            have_target = True

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
                (start - offset, i - offset)
            )  # 因为是切片，i-1实际上会取到i-2范围,logits的核心就是不要预测任何-100
            start = None

    if start is not None:
        # 这个地方结束位置一般不重要，除非最后有什么不需要预测的特殊标志。
        # 到底什么情况会进这里呢？整句话都不存在-100的时候
        if have_target:
            ranges.append((start - offset, len(lst) - offset))
        else:
            ranges.append((0, len(lst) - offset))

    return ranges


class CompressedCounter:
    def __init__(self):
        self.counts = {}  # 只存储非零计数
        self.num_values = 0

    def add(self, value):
        if value in self.counts:
            self.counts[value] += 1
        else:
            self.counts[value] = 1
            self.num_values += 1

    def get(self, value):
        return self.counts.get(value, 0)

    def merge(self, other):
        for value, count in other.counts.items():
            if value in self.counts:
                self.counts[value] += count
            else:
                self.counts[value] = count
                self.num_values += 1

    def items(self):
        return self.counts.items()

    def __len__(self):
        return self.num_values

    def __str__(self):
        return f"{self.counts}"

    def __repr__(self):
        return self.__str__()


class TrieNode:
    def __init__(self):
        self.children = {}
        self._counts = None  # 懒初始化

    def add_value(self, value):
        if self._counts is None:
            self._counts = CompressedCounter()
        self._counts.add(value)

    def get_counts(self):
        return self._counts

    def __str__(self):
        return f"self.children:{self.children.keys()}\n"

    def __repr__(self):
        return self.__str__()


class SuffixTrie:
    def __init__(self):
        self.root = TrieNode()

    def _insert(self, input_id: list[int], label: list[int]):
        """_summary_

        Args:
            input_id (list[int]): _description_
            label (list[int]): _description_
            window_size (_type_, optional): Defaults to infinite window size.
        """
        node = self.root

        for i in range(len(input_id) - 1, -1, -1):

            # if input_id[i]==-100:
            #     # clm 会碰到这种情况
            #     continue
            if input_id[i] not in node.children:
                node.children[input_id[i]] = TrieNode()
            node = node.children[input_id[i]]

            node.add_value(label[-1])

    def insert(self, input_id: list[int], label: list[int], window_size=math.inf):

        non_zero_indices = find_ranges(label)
        for start, end in non_zero_indices:
            for i in range(start, end):
                self._insert(
                    input_id[i - window_size : i],
                    label[i - window_size + 1 : i + 1],
                )

    def search(self, key_list, backoff_step):

        counter_list = []
        node = self.root
        last_not_none_idx = -1
        for idx, key in enumerate(reversed(key_list)):
            if key in node.children:
                node = node.children[key]
                if node._counts == None:

                    counter_list.append(CompressedCounter())
                else:
                    counter_list.append(node._counts)
                    last_not_none_idx = idx
                # print(idx,key)
            else:
                break
        # 左侧的解释：有可能有时候backoff_step是0，就会把全部返回，这时候应该返回一个就很好
        # 右侧的解释，有可能需要退避很多步才能找到，这在探测边界会出现


        return counter_list[
            last_not_none_idx + 1 - max(backoff_step, 1) : last_not_none_idx + 1
        ]

    def __str__(self):
        return self._print_tree(self.root)

    def __repr__(self):
        return self.__str__()

    def _print_tree(self, node, level=0, prefix="Root"):
        """递归打印树的结构，包括计数信息"""
        # 获取计数信息的字符串表示
        counts_str = ""
        if node._counts:
            # 假设 _counts 有一个 items() 方法返回计数键值对
            counts_items = (
                node._counts.items()
            )  # 或者根据你的 CompressedCounter 实现调整
            counts_str = (
                " Counts{" + ", ".join(f"{k}:{v}" for k, v in counts_items) + "}"
            )

        # 打印当前节点信息
        result = "  " * level + f"{prefix}{counts_str}\n"

        # 递归打印子节点
        for key, child in node.children.items():
            result += self._print_tree(child, level + 1, f"└─ {key}")
        return result


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma_2b", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--template", type=str)
    parser.add_argument("--mono", default=False, type=ast.literal_eval)
    parser.add_argument("--mono_dataset", default="wiki_medical", type=str)
    parser.add_argument("--window-size", default=256, type=int)
    parser.add_argument("--backoff-ratio", default=0, type=float)
    return parser.parse_args()


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
    """将数据分块保存,同时保存元数据"""
    metadata = {
        "chunk_size": chunk_size,
        "total_size": len(data),
        "num_chunks": math.ceil(len(data) / chunk_size),
    }

    # 保存元数据
    with open(os.path.join(base_dir, f"{name}_metadata.json"), "w") as f:
        json.dump(metadata, f)

    # 保存数据块
    for i, chunk in enumerate(chunk_data(data, chunk_size)):
        filename = f"{name}_part{i+start_idx}.pkl"
        with open(os.path.join(base_dir, filename), "wb") as f:
            pickle.dump(chunk, f, protocol=pickle.HIGHEST_PROTOCOL)


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
                mode=1,
                test=False,
            ),
            batched=True,
            num_proc=30,
            remove_columns=train_dataset.features.keys(),
            # load_from_cache_file=False,
            desc="tokenize",
        )
        dataset_list.append(train_dataset)
    return datasets.concatenate_datasets(dataset_list)


def statistic(args, train_dataset, mono_dataset=None):

    trie = SuffixTrie()

    logger.debug(f"start to make statistic")
    # 统计begin-----------------------------------------------------------------------------------
    for i in tqdm(range(len(train_dataset)), desc="statistic stage"):

        input_id, label = train_dataset[i]["input_ids"], train_dataset[i]["labels"]
        assert len(input_id) == len(label)
        trie.insert(input_id, label, window_size=args.window_size)

    if args.mono:
        for i in tqdm(range(len(mono_dataset)), desc="mono statistic stage"):
            input_id, label = (
                mono_dataset[i]["input_ids"],
                mono_dataset[i]["labels"],
            )

            trie.insert(input_id, label, window_size=args.window_size)

    return trie


def synthesis(args, train_dataset, trie: SuffixTrie, template):
    tokenizer = template.tokenizer  # debug usage
    synthesis_dict = defaultdict(list)
    cnt_list = []  # list[list[tuple]] sentence_idx, -100 area , (start,end)

    for i in tqdm(range(len(train_dataset)), desc="synthesis stage"):

        input_id, label = train_dataset[i]["input_ids"], train_dataset[i]["labels"]

        # 这个地方和encoder-decoder模型还不一样，不需要特地区分编解码的输入，所以只需要一个input_id即可，input_id最后的EOS不需要送给模型
        key = tuple(input_id[:-1])
        length = len(input_id)
        if synthesis_dict[key] == []:
            #     or (
            #     template.chat_eos_token_id not in synthesis_dict[key][-1][0]
            #     and template.chat_eos_token_id
            #     not in synthesis_dict[key][-2][
            #         0
            #     ]  # qwen/gemma的template结尾是\n，所以还需要再检查一下-2位置
            # )
            # 防止重复示例. 情况1，这条数据没有被添加过了，情况2，这条数据没有被添加到结束符
            # cnt list必须在这里，不然对synthesis_dict的去重会导致长度不匹配
            ranges_list = find_ranges(label, offset=1)
            cnt_list.append(ranges_list)

            for ranges in ranges_list:
                for i in range(ranges[0], ranges[1]):
                    counter_list = trie.search(
                        key[max(0, i + 1 - args.window_size) : i + 1],
                        math.floor(args.window_size * args.backoff_ratio),
                    )
                    if counter_list==[]:
                        merged_counter=CompressedCounter()
                    else:
                        merged_counter = counter_list[0]
                        for counter in counter_list[1:]:
                            merged_counter.merge(counter)
                    synthesis_dict[key].append(merged_counter)
    return synthesis_dict, cnt_list


def create_save_path(args):
    """
    根据参数创建唯一的保存路径

    Args:
        args: 解析的命令行参数
    Returns:
        str: 生成的完整保存路径
    """
    # 获取脚本所在目录的绝对路径
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 处理模型名称，提取纯模型名
    model_name = os.path.basename(args.model)

    # 构建路径组件
    components = [
        base_dir,
        "train_dataset",
        f"{model_name}",
        f"{args.template}_{args.dataset}",
    ]

    # 添加可选组件
    if args.mono:
        components.append(f"mono_{args.mono_dataset.replace(',', '_')}")

    # 添加窗口大小和退避率参数
    components.append(f"w{args.window_size}_b{args.backoff_ratio}")

    # 组合路径
    save_path = os.path.join(*components)

    # 确保目录存在
    os.makedirs(save_path, exist_ok=True)

    return save_path


@logger.catch
def main():

    args = parse_args()

    # while True:
    #     a=str(input('你要跳过检查吗？y/n'))
    #     if a=='y':
    #         break
    #     elif a=='n':
    #         test(args)

    #         break

    tokenizer = AutoTokenizer.from_pretrained(model_dir.get(args.model, args.model))
    # config = AutoConfig.from_pretrained(model_dir[args.model])
    # model_type = config.model_type
    # template = modelType2Template[model_type](tokenizer)
    template = modelType2Template[args.template](tokenizer)

    train_dataset = parse_dataset(args, template, args.dataset)
    # train_dataset = train_dataset.sort('input_ids')
    if args.mono:
        mono_dataset = parse_dataset(args, template, args.mono_dataset)
        trie = statistic(args, train_dataset, mono_dataset)
    else:
        trie = statistic(args, train_dataset)

    synthesis_dict, cnt_list = synthesis(args, train_dataset, trie, template)

    logger.debug(
        f"length of synthesis_dict:{len(synthesis_dict)};length of cnt_list:{len(cnt_list)}"
    )
    assert len(synthesis_dict) == len(cnt_list)

    output_base_dir = create_save_path(args)

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


def test(args):
    tokenizer = AutoTokenizer.from_pretrained(model_dir.get("gemma_2b"))
    template = modelType2Template["gemma"](tokenizer)
    mock_dataset = dname2load["test"](None)
    mono_dataset = parse_dataset(args, template, "test")
    mock_dataset = mock_dataset.map(
        partial(
            dname2func["test"],
            template=template,
            mode=1,
            test=False,
        ),
        batched=True,
        num_proc=30,
        # remove_columns=train_dataset.features.keys(),
        load_from_cache_file=False,
        desc="tokenize",
    )
    supervised_trie, clm_trie = statistic(args, mock_dataset, mono_dataset)
    if args.template == "qwen2":
        if args.ngram == 0:
            target_synthesis_dict = {
                (
                    2,
                    106,
                    1645,
                    108,
                    235285,
                    2182,
                    692,
                    604,
                    2149,
                    13669,
                    1069,
                    107,
                    108,
                    106,
                    2516,
                    108,
                    235285,
                    2182,
                    692,
                    604,
                    2149,
                    13669,
                    1069,
                    1,
                ): [
                    [Counter({235285: 1}), Counter({235285: 4})],
                    [Counter({2182: 1}), Counter({2182: 2, 798: 2})],
                    [Counter({692: 1}), Counter({692: 2})],
                    [Counter({604: 1}), Counter({604: 2})],
                    [Counter({2149: 1}), Counter({2149: 2})],
                    [Counter({13669: 1}), Counter({13669: 2})],
                    [Counter({1069: 1}), Counter({1069: 2})],
                    [Counter({1: 1}), Counter({1: 2})],
                    [Counter({108: 1}), Counter({108: 2})],
                ],
                (
                    2,
                    106,
                    1645,
                    108,
                    235285,
                    798,
                    235303,
                    235251,
                    1707,
                    692,
                    107,
                    108,
                    106,
                    2516,
                    108,
                    235285,
                    798,
                    235303,
                    235251,
                    1707,
                    692,
                    1,
                ): [
                    [Counter({235285: 1}), Counter({235285: 4})],
                    [Counter({798: 1}), Counter({2182: 2, 798: 2})],
                    [Counter({235303: 1}), Counter({235303: 2})],
                    [Counter({235251: 1}), Counter({235251: 2})],
                    [Counter({1707: 1}), Counter({1707: 2})],
                    [Counter({692: 1}), Counter({692: 2})],
                    [Counter({1: 1}), Counter({1: 2})],
                    [Counter({108: 1}), Counter({108: 2})],
                ],
            }
        elif args.ngram == 4:
            target_synthesis_dict = {
                (
                    2,
                    106,
                    1645,
                    108,
                    235285,
                    2182,
                    692,
                    604,
                    2149,
                    13669,
                    1069,
                    107,
                    108,
                    106,
                    2516,
                    108,
                    235285,
                    2182,
                    692,
                    604,
                    2149,
                    13669,
                    1069,
                    1,
                ): [
                    [Counter({235285: 1}), Counter({235285: 1})],
                    [Counter({2182: 1}), Counter({2182: 1})],
                    [Counter({692: 1}), Counter({692: 1})],
                    [Counter({604: 1}), Counter({604: 1})],
                    [Counter({2149: 1}), Counter({2149: 2})],
                    [Counter({13669: 1}), Counter({13669: 2})],
                    [Counter({1069: 1}), Counter({1069: 2})],
                    [Counter({1: 1}), Counter({1: 2})],
                    [Counter({108: 1}), Counter({108: 2})],
                ],
                (
                    2,
                    106,
                    1645,
                    108,
                    235285,
                    798,
                    235303,
                    235251,
                    1707,
                    692,
                    107,
                    108,
                    106,
                    2516,
                    108,
                    235285,
                    798,
                    235303,
                    235251,
                    1707,
                    692,
                    1,
                ): [
                    [Counter({235285: 1}), Counter({235285: 1})],
                    [Counter({798: 1}), Counter({798: 1})],
                    [Counter({235303: 1}), Counter({235303: 1})],
                    [Counter({235251: 1}), Counter({235251: 1})],
                    [Counter({1707: 1}), Counter({1707: 2})],
                    [Counter({692: 1}), Counter({692: 2})],
                    [Counter({1: 1}), Counter({1: 2})],
                    [Counter({108: 1}), Counter({108: 2})],
                ],
            }
        elif args.ngram == 1:
            target_synthesis_dict = {
                (
                    2,
                    106,
                    1645,
                    108,
                    235285,
                    2182,
                    692,
                    604,
                    2149,
                    13669,
                    1069,
                    107,
                    108,
                    106,
                    2516,
                    108,
                    235285,
                    2182,
                    692,
                    604,
                    2149,
                    13669,
                    1069,
                    1,
                ): [
                    [Counter({235285: 1}), Counter({235285: 1})],
                    [Counter({2182: 1}), Counter({2182: 2, 798: 2})],
                    [Counter({692: 1}), Counter({692: 2})],
                    [Counter({604: 1}), Counter({604: 2})],
                    [Counter({2149: 1}), Counter({2149: 2})],
                    [Counter({13669: 1}), Counter({13669: 2})],
                    [Counter({1069: 1}), Counter({1069: 2})],
                    [Counter({1: 1}), Counter({1: 2})],
                    [Counter({108: 1}), Counter({108: 2})],
                ],
                (
                    2,
                    106,
                    1645,
                    108,
                    235285,
                    798,
                    235303,
                    235251,
                    1707,
                    692,
                    107,
                    108,
                    106,
                    2516,
                    108,
                    235285,
                    798,
                    235303,
                    235251,
                    1707,
                    692,
                    1,
                ): [
                    [Counter({235285: 1}), Counter({235285: 1})],
                    [Counter({798: 1}), Counter({2182: 2, 798: 2})],
                    [Counter({235303: 1}), Counter({235303: 2})],
                    [Counter({235251: 1}), Counter({235251: 2})],
                    [Counter({1707: 1}), Counter({1707: 2})],
                    [Counter({692: 1}), Counter({692: 2})],
                    [Counter({1: 1}), Counter({1: 2})],
                    [Counter({108: 1}), Counter({108: 2})],
                ],
            }
    else:
        if args.ngram == 1:
            target_synthesis_dict = {
                (
                    2,
                    106,
                    1645,
                    108,
                    235285,
                    2182,
                    692,
                    604,
                    2149,
                    13669,
                    1069,
                    107,
                    108,
                    106,
                    2516,
                    108,
                    235285,
                    2182,
                    692,
                    604,
                    2149,
                    13669,
                    1069,
                    1,
                ): [
                    [Counter({235285: 1}), Counter({235285: 1})],
                    [Counter({2182: 1}), Counter({2182: 2, 798: 2})],
                    [Counter({692: 1}), Counter({692: 2})],
                    [Counter({604: 1}), Counter({604: 2})],
                    [Counter({2149: 1}), Counter({2149: 2})],
                    [Counter({13669: 1}), Counter({13669: 2})],
                    [Counter({1069: 1}), Counter({1069: 2})],
                    [Counter({1: 1}), Counter({1: 2})],
                    [Counter({108: 1}), Counter({108: 2})],
                ],
                (
                    2,
                    106,
                    1645,
                    108,
                    235285,
                    798,
                    235303,
                    235251,
                    1707,
                    692,
                    107,
                    108,
                    106,
                    2516,
                    108,
                    235285,
                    798,
                    235303,
                    235251,
                    1707,
                    692,
                    1,
                ): [
                    [Counter({235285: 1}), Counter({235285: 1})],
                    [Counter({798: 1}), Counter({2182: 2, 798: 2})],
                    [Counter({235303: 1}), Counter({235303: 2})],
                    [Counter({235251: 1}), Counter({235251: 2})],
                    [Counter({1707: 1}), Counter({1707: 2})],
                    [Counter({692: 1}), Counter({692: 2})],
                    [Counter({1: 1}), Counter({1: 2})],
                    [Counter({108: 1}), Counter({108: 2})],
                ],
            }
        elif args.ngram == 4:
            target_synthesis_dict = {
                (
                    2,
                    106,
                    1645,
                    108,
                    235285,
                    2182,
                    692,
                    604,
                    2149,
                    13669,
                    1069,
                    107,
                    108,
                    106,
                    2516,
                    108,
                    235285,
                    2182,
                    692,
                    604,
                    2149,
                    13669,
                    1069,
                    1,
                ): [
                    [Counter({235285: 1}), Counter({235285: 1})],
                    [Counter({2182: 1}), Counter({2182: 1, 798: 1})],
                    [Counter({692: 1}), Counter({692: 1})],
                    [Counter({604: 1}), Counter({604: 1})],
                    [Counter({2149: 1}), Counter({2149: 1})],
                    [Counter({13669: 1}), Counter({13669: 1})],
                    [Counter({1069: 1}), Counter({1069: 1})],
                    [Counter({1: 1}), Counter({1: 1})],
                    [Counter({108: 1}), Counter({108: 1})],
                ],
                (
                    2,
                    106,
                    1645,
                    108,
                    235285,
                    798,
                    235303,
                    235251,
                    1707,
                    692,
                    107,
                    108,
                    106,
                    2516,
                    108,
                    235285,
                    798,
                    235303,
                    235251,
                    1707,
                    692,
                    1,
                ): [
                    [Counter({235285: 1}), Counter({235285: 1})],
                    [Counter({798: 1}), Counter({2182: 1, 798: 1})],
                    [Counter({235303: 1}), Counter({235303: 1})],
                    [Counter({235251: 1}), Counter({235251: 1})],
                    [Counter({1707: 1}), Counter({1707: 1})],
                    [Counter({692: 1}), Counter({692: 1})],
                    [Counter({1: 1}), Counter({1: 1})],
                    [Counter({108: 1}), Counter({108: 1})],
                ],
            }
    target_cnt_list = [[(15, 24)], [(14, 22)]]
    synthesis_dict, cnt_list = synthesis(
        args, mock_dataset, supervised_trie, clm_trie, template
    )

    if target_synthesis_dict != synthesis_dict:
        exit()

    # assert target_synthesis_dict==synthesis_dict
    # assert cnt_list == target_cnt_list


if __name__ == "__main__":
    main()

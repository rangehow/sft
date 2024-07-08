from functools import partial
import os
from collections import Counter, defaultdict
from transformers import AutoTokenizer, AutoConfig
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
from multiprocessing import Pool, cpu_count
warnings.filterwarnings("ignore", "The iteration is not making good progress")


class TrieNode:
    def __init__(self):
        self.children = {}
        self.value = Counter()


class Trie:
    def __init__(self):
        self.root = TrieNode()

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma_2b")
    parser.add_argument("--dataset", default="alpaca_cleaned")
    parser.add_argument("--clm", default=True, type=ast.literal_eval)
    parser.add_argument("--ngram", default=4)
    parser.add_argument("--cache_statistic", default=True, type=ast.literal_eval)
    parser.add_argument("--template", type=str)
    return parser.parse_args()


def find_ranges(lst, target=-100):
    ranges = []
    start = None
    multiTurnOnlyOnceInfoFlag = True
    for i, num in enumerate(lst):
        if num != target and start is None:
            start = i
        elif num == target and start is not None:
            if multiTurnOnlyOnceInfoFlag:
                logger.info(
                    "这个分支理论上只有多轮对话的数据集才会进入,确保自己在使用多轮对话数据集"
                )
                multiTurnOnlyOnceInfoFlag = False
            ranges.append(
                (start - 1, i - 1)
            )
            start = None

    if start is not None:
        ranges.append((start - 1, len(lst) - 1))

    return ranges


from itertools import islice


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


def save_chunks(data, chunk_size, base_dir, name):
    """将大字典分块并保存到多个文件中。"""
    for i, chunk in enumerate(chunk_data(data, chunk_size)):
        filename = f"{name}_part{i}.msgpack"
        with open(os.path.join(base_dir, filename), "wb") as f:
            pickle.dump(chunk, f, protocol=5)
        print(f"Saved chunk {i} to {filename}")


def process_statistic_chunk(args, clm, ngram, train_dataset_chunk):
    supervised_trie = Trie()
    clm_trie = Trie() if clm else None

    for data in train_dataset_chunk:
        input_id, label = data["input_ids"], data["labels"]

        if clm:
            flag4LossArea = False
            regionBeginIdx = -1

        for i in range(len(label) - 1):
            if label[i + 1] != -100:
                supervised_trie.insert(input_id[: i + 1], label[i + 1])

                if clm:
                    if flag4LossArea is False:
                        regionBeginIdx = i
                        flag4LossArea = True

                    if i - regionBeginIdx >= ngram:
                        clm_trie.insert(label[regionBeginIdx + 1 : i + 1], label[i + 1])
            elif clm and flag4LossArea:
                flag4LossArea = False

    return supervised_trie, clm_trie


def statistic(train_dataset, clm, ngram):
    chunk_size = len(train_dataset) // cpu_count()
    chunks = list(chunk_data(train_dataset, chunk_size))

    args = parse_args()
    with Pool(cpu_count()) as pool:
        results = pool.starmap(
            process_statistic_chunk,
            [(args, clm, ngram, chunk) for chunk in chunks]
        )

    supervised_trie = Trie()
    clm_trie = Trie() if clm else None

    for s_trie, c_trie in results:
        merge_tries(supervised_trie, s_trie)
        if clm and c_trie:
            merge_tries(clm_trie, c_trie)

    return supervised_trie, clm_trie


def merge_tries(main_trie, new_trie):
    stack = [(main_trie.root, new_trie.root)]
    while stack:
        main_node, new_node = stack.pop()
        for key, new_child in new_node.children.items():
            if key not in main_node.children:
                main_node.children[key] = new_child
            else:
                main_node.children[key].value.update(new_child.value)
                stack.append((main_node.children[key], new_child))


def process_synthesis_chunk(args, tokenizer, supervised_trie, clm_trie, train_dataset_chunk):
    synthesis_dict = defaultdict(list)
    cnt_list = []

    for data in train_dataset_chunk:
        input_id, label = data["input_ids"], data["labels"]

        if args.clm:
            flag4LossArea = False
            regionBeginIdx = -1

        key = tuple(input_id[:-1])
        length = len(input_id)
        if synthesis_dict[key] == [] or (
            tokenizer.eos_token_id not in synthesis_dict[key][-1][0]
            and tokenizer.eos_token_id
            not in synthesis_dict[key][-2][0]
        ):
            cnt_list.append(find_ranges(label))

            for i in range(length - 1):
                if label[i + 1] != -100:
                    supervised_value = supervised_trie.search(input_id[: i + 1])

                    if args.clm:
                        if flag4LossArea is False:
                            regionBeginIdx = i
                            flag4LossArea = True

                        clm_value = clm_trie.search(label[regionBeginIdx + 1 : i + 1])
                        if clm_value is None or len(clm_value) == 0:
                            clm_value = supervised_value

                    synthesis_dict[key].append([supervised_value, clm_value])

                elif args.clm and flag4LossArea:
                    flag4LossArea = False

    return synthesis_dict, cnt_list


def synthesis(train_dataset, tokenizer, supervised_trie, clm_trie):
    chunk_size = len(train_dataset) // cpu_count()
    chunks = list(chunk_data(train_dataset, chunk_size))

    args = parse_args()
    with Pool(cpu_count()) as pool:
        results = pool.starmap(
            process_synthesis_chunk,
            [(args, tokenizer, supervised_trie, clm_trie, chunk) for chunk in chunks]
        )

    synthesis_dict = defaultdict(list)
    cnt_list = []

    for s_dict, c_list in results:
        for k, v in s_dict.items():
            synthesis_dict[k].extend(v)
        cnt_list.extend(c_list)

    return synthesis_dict, cnt_list


@logger.catch
def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(model_dir[args.model])
    template = modelType2Template[args.template](tokenizer)
    dataset_name_list = args.dataset.split(",")
    dataset_list = []
    for dname in dataset_name_list:
        train_dataset = dname2load[dname](dataset_dir.get(dname, None))

        train_dataset = train_dataset.map(
            partial(dname2func[dname], template=template, mode=1, test=False),
            batched=True,
            num_proc=30,
            load_from_cache_file=False,
            desc="tokenize",
        )
        dataset_list.append(train_dataset)
    train_dataset = datasets.concatenate_datasets(dataset_list)

    supervised_trie, clm_trie = statistic(train_dataset, args.clm, args.ngram)
    synthesis_dict, cnt_list = synthesis(train_dataset, tokenizer, supervised_trie, clm_trie)

    logger.debug(
        f"length of synthesis_dict:{len(synthesis_dict)};length of cnt_list:{len(cnt_list)}"
    )
    assert len(synthesis_dict) == len(cnt_list)

    script_path = os.path.dirname(os.path.abspath(__file__).rstrip(os.sep))
    os.makedirs(
        os.path.join(script_path, "train_dataset", f"{args.template}_{args.dataset}"),
        exist_ok=True,
    )

    save_chunks(
        synthesis_dict,
        chunk_size=500,
        base_dir=f"{script_path}/train_dataset/{args.template}_{args.dataset}",
        name="synthesis",
    )
    save_chunks(
        cnt_list,
        chunk_size=500,
        base_dir=f"{script_path}/train_dataset/{args.template}_{args.dataset}",
        name="index",
    )

    logger.debug(f"整合文件被保存到train_dataset/{args.template}_{args.dataset}")


def load_msgpack_file(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def find_msgpack_chunk_files(base_dir, name):
    """查找与基准文件名匹配的所有 msgpack 分块文件。"""
    chunk_files = [
        os.path.join(base_dir, f)
        for f in os.listdir(base_dir)
        if f.startswith(name) and f.endswith(".msgpack")
    ]
    return sorted(chunk_files)


import concurrent.futures


def load_msgpack_chunks(chunk_files):
    print(chunk_files)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(load_msgpack_file, chunk_files))
    if isinstance(results[0], dict):
        merged_data = {}
        for chunk in results:
            merged_data.update(chunk)
        return merged_data
    elif isinstance(results[0], list):
        merged_data = []
        for chunk in results:
            merged_data.extend(chunk)
        return merged_data
    else:
        raise TypeError("data must be a dictionary or a list")


def test():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(model_dir[args.model])
    template = modelType2Template[args.template](tokenizer)

    script_path = os.path.dirname(os.path.abspath(__file__).rstrip(os.sep))
    base_dir = f"{script_path}/train_dataset/{args.template}_{args.dataset}"
    synthesis_dict = load_msgpack_chunks(
        find_msgpack_chunk_files(base_dir, name="synthesis")
    )
    cnt_list = load_msgpack_chunks(find_msgpack_chunk_files(base_dir, name="index"))

    import pdb
    pdb.set_trace()

    train_dataset = datasets.load_dataset(dataset_dir[args.dataset])["train"]

    train_dataset = train_dataset.map(
        partial(dname2func[args.dataset], template=template),
        batched=True,
        num_proc=30,
        remove_columns=train_dataset.features.keys(),
        desc="tokenize",
    )

    synthesis_dict = [data_sample for data_sample in synthesis_dict.items()]

    cnt = 0
    for i in range(len(synthesis_dict)):
        input_ids = synthesis_dict[i][0]
        length = cnt_list[i][0][-1]
        if len(input_ids) != length:
            pdb.set_trace()

    logger.debug(len(synthesis_dict))
    logger.debug(len(train_dataset))


if __name__ == "__main__":
    main()
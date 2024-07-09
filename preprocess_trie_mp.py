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
import concurrent.futures

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

    def merge(self, other):
        def _merge_nodes(node1, node2):
            node1.value.update(node2.value)
            for key, child_node2 in node2.children.items():
                if key in node1.children:
                    _merge_nodes(node1.children[key], child_node2)
                else:
                    node1.children[key] = child_node2

        _merge_nodes(self.root, other.root)


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
            ranges.append((start - 1, i - 1))
            start = None

    if start is not None:
        ranges.append((start - 1, len(lst) - 1))

    return ranges


from itertools import islice


def chunk_data(data, chunk_size):
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
    for i, chunk in enumerate(chunk_data(data, chunk_size)):
        filename = f"{name}_part{i}.msgpack"
        with open(os.path.join(base_dir, filename), "wb") as f:
            pickle.dump(chunk, f, protocol=5)
        print(f"Saved chunk {i} to {filename}")


def process_dataset_chunk(chunk, args):
    supervised_trie = Trie()
    clm_trie = Trie() if args.clm else None

    for c in chunk:
        input_id, label = c[0], c[1]

        if args.clm:
            flag4LossArea = False
            regionBeginIdx = -1

        for i in range(len(label) - 1):
            if label[i + 1] != -100:
                supervised_trie.insert(input_id[: i + 1], label[i + 1])
                if args.clm:
                    if flag4LossArea is False:
                        regionBeginIdx = i
                        flag4LossArea = True
                    if i - regionBeginIdx >= args.ngram:
                        clm_trie.insert(label[regionBeginIdx + 1 : i + 1], label[i + 1])
            elif args.clm and flag4LossArea:
                flag4LossArea = False

    return supervised_trie, clm_trie


def statistic_concurrently(train_dataset, args):
    chunk_size = 1000
    chunks = []
    for i, c in enumerate(train_dataset):
        temp_data = train_dataset[i : i + chunk_size]
        chunks.append((temp_data["input_ids"], temp_data["labels"]))
    # 示例数据集

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_dataset_chunk, chunk, args) for chunk in chunks
        ]
        results = [
            f.result()
            for f in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="statistic stage",
            )
        ]

    supervised_trie = Trie()
    clm_trie = Trie() if args.clm else None

    for sup_trie, clm_trie_part in results:
        supervised_trie.merge(sup_trie)
        if args.clm:
            clm_trie.merge(clm_trie_part)

    return supervised_trie, clm_trie


def process_synthesis_chunk(
    chunk, cnt_list, args, supervised_trie, clm_trie, tokenizer
):
    synthesis_dict = defaultdict(list)
    for item, ranges in zip(chunk, cnt_list):
        input_id, label = item["input_ids"], item["labels"]
        if args.clm:
            flag4LossArea = False
            regionBeginIdx = -1

        key = tuple(input_id[:-1])
        length = len(input_id)
        if synthesis_dict[key] == [] or (
            tokenizer.eos_token_id not in synthesis_dict[key][-1][0]
            and tokenizer.eos_token_id not in synthesis_dict[key][-2][0]
        ):
            for i in range(length - 1):
                if label[i + 1] != -100:
                    supervised_value = supervised_trie.search(input_id[: i + 1])
                    clm_value = (
                        clm_trie.search(label[regionBeginIdx + 1 : i + 1])
                        if args.clm
                        else None
                    )
                    if clm_value is None or len(clm_value) == 0:
                        clm_value = supervised_value
                    synthesis_dict[key].append([supervised_value, clm_value])
                elif args.clm and flag4LossArea:
                    flag4LossArea = False
    return synthesis_dict


def synthesis_concurrently(
    train_dataset, cnt_list, args, supervised_trie, clm_trie, tokenizer
):
    chunk_size = 1000
    chunks = list(chunk_data(train_dataset, chunk_size))
    cnt_chunks = list(chunk_data(cnt_list, chunk_size))

    with concurrent.futures.ProcessPoolExecutor() as executor:

        futures = [
            executor.submit(
                process_synthesis_chunk,
                chunk,
                cnt_chunk,
                args,
                supervised_trie,
                clm_trie,
                tokenizer,
            )
            for chunk, cnt_chunk in zip(chunks, cnt_chunks)
        ]
        results = [
            f.result()
            for f in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="synthesis stage",
            )
        ]

    synthesis_dict = defaultdict(list)
    for part in results:
        for key, value in part.items():
            synthesis_dict[key].extend(value)

    return synthesis_dict


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

    supervised_trie, clm_trie = statistic_concurrently(train_dataset, args)

    cnt_list = []
    for item in train_dataset:
        cnt_list.append(find_ranges(item["labels"]))

    synthesis_dict = synthesis_concurrently(
        train_dataset, cnt_list, args, supervised_trie, clm_trie, tokenizer
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
    chunk_files = [
        os.path.join(base_dir, f)
        for f in os.listdir(base_dir)
        if f.startswith(name) and f.endswith(".msgpack")
    ]
    return sorted(chunk_files)


def load_msgpack_chunks(chunk_files):
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

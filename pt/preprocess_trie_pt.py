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
            # Merge the values
            node1.value.update(node2.value)
            # Merge the children
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
    parser.add_argument("--ngram", default=0)
    parser.add_argument("--cache_statistic", default=True, type=ast.literal_eval)
    parser.add_argument("--template", type=str)

    return parser.parse_args()



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


@logger.catch
def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(model_dir[args.model])
    # config = AutoConfig.from_pretrained(model_dir[args.model])
    # model_type = config.model_type
    # template = modelType2Template[model_type](tokenizer)
    template = modelType2Template[args.template](tokenizer)
    dataset_name_list = args.dataset.split(",")
    dataset_list = []
    for dname in dataset_name_list:
        train_dataset = dname2load[dname](dataset_dir.get(dname, None))
        # train_dataset = datasets.load_dataset(dataset_dir.get(args.dataset, args.dataset))[
        #     "train"
        # ]

        print("\n数据集", dname, "=")
        print(train_dataset)
        train_dataset = train_dataset.map(
            partial(dname2func[dname], template=template, mode=1, test=False),
            batched=True,
            num_proc=30,
            # remove_columns=train_dataset.features.keys(),
            load_from_cache_file=False,
            desc="tokenize",
        )
        dataset_list.append(train_dataset)
    train_dataset = datasets.concatenate_datasets(dataset_list)
    # train_dataset = train_dataset.sort('input_ids')

    def statistic():
        
        clm_trie = Trie()

        logger.debug(f"start to make statistic")
        # 统计begin-----------------------------------------------------------------------------------
        for j in tqdm(range(len(train_dataset)), desc="statistic stage"):

            label = train_dataset[j]["labels"]

            for i in range(len(label) - 1):
                    
                clm_trie.insert(
                    label[: i + 1], label[i + 1]
                )
                
        return clm_trie

    clm_trie = statistic()

    def synthesis():
        synthesis_dict = defaultdict(list)
        cnt_list = []

        for j in tqdm(range(len(train_dataset)), desc="synthesis stage"):

            label =  train_dataset[j]["labels"]

            # 这个地方和encoder-decoder模型还不一样，不需要特地区分编解码的输入，所以只需要一个input_id即可，input_id最后的EOS不需要送给模型
            key = tuple(label[:-1])
            length = len(label)
            if synthesis_dict[key] == [] or (
                tokenizer.eos_token_id not in synthesis_dict[key][-1][0]
                and tokenizer.eos_token_id
                not in synthesis_dict[key][-2][
                    0
                ]  # qwen的template结尾是\n，我无语了。。
            ):  # 防止重复示例. 情况1，这条数据没有被添加过了，情况2，这条数据没有被添加到结束符
                # cnt list必须在这里，不然对synthesis_dict的去重会导致长度不匹配
                cnt_list.append([(0,length-1)])

                for i in range(
                    length - 1
                ):  
  
                    clm_value = clm_trie.search(
                        label[: i + 1]
                    )

                    synthesis_dict[key].append([clm_value])


                    if len(synthesis_dict) != len(cnt_list):
                        # llama3-base和it的eos token不一样
                        import pdb

                        pdb.set_trace()

        return synthesis_dict, cnt_list

    synthesis_dict, cnt_list = synthesis()

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
        return pickle.load(f)  # ,strict_map_key=False,strict_types =True


def find_msgpack_chunk_files(
    base_dir,
    name,
):
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

    # 示例使用
    script_path = os.path.dirname(os.path.abspath(__file__).rstrip(os.sep))
    base_dir = f"{script_path}/train_dataset/{args.template}_{args.dataset}"
    synthesis_dict = load_msgpack_chunks(
        find_msgpack_chunk_files(base_dir, name="synthesis")
    )
    cnt_list = load_msgpack_chunks(find_msgpack_chunk_files(base_dir, name="index"))



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
        # logger.debug(train_dataset[cnt])

        # cnt+=1
    logger.debug(len(synthesis_dict))
    logger.debug(len(train_dataset))


if __name__ == "__main__":
    main()

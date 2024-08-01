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
from concurrent.futures import ThreadPoolExecutor

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

    def __str__(self):
        def _print(node, prefix):
            if node.value:
                print(f"{prefix}: {dict(node.value)}")
            for key, child in node.children.items():
                _print(child, prefix + [key])

        _print(self.root, [])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma_2b")
    parser.add_argument("--dataset")
    parser.add_argument("--clm", default=True, type=ast.literal_eval)
    parser.add_argument("--ngram", default=4, type=int)
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
        if num == target:
            have_target = True

        if num != target and start is None:
            start = i
        elif num == target and start is not None:
            if multiTurnOnlyOnceInfoFlag:
                logger.info("这个分支理论上只有多轮对话的数据集才会进入,确保自己在使用多轮对话数据集")
                multiTurnOnlyOnceInfoFlag = False
            ranges.append((start - 1, i - 1))
            start = None

    if start is not None:
        if have_target:
            ranges.append((start - 1, len(lst) - 1))
        else:
            ranges.append((0, len(lst) - 1))

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


def save_chunks(data, chunk_size, base_dir, name, start_idx=0):
    for i, chunk in tqdm(enumerate(chunk_data(data, chunk_size))):
        filename = f"{name}_part{i+start_idx}.msgpack"
        with open(os.path.join(base_dir, filename), "wb") as f:
            pickle.dump(chunk, f, protocol=5)
        print(f"Saved chunk {i} to {filename}")


def parse_dataset(args, template, dataset_str):
    dataset_name_list = dataset_str.split(",")
    dataset_list = []
    for dname in dataset_name_list:
        train_dataset = dname2load[dname](dataset_dir.get(dname, None))
        print("\n数据集", dname, "=")
        print(train_dataset)
        train_dataset = train_dataset.map(
            partial(dname2func[dname], template=template, mode=1 if args.w_template else 0, test=False),
            batched=True,
            num_proc=30,
            remove_columns=train_dataset.features.keys(),
            load_from_cache_file=False,
            desc="tokenize",
        )
        dataset_list.append(train_dataset)
    return datasets.concatenate_datasets(dataset_list)


def split_dataset(dataset, num_splits):
    split_size = len(dataset) // num_splits
    return [dataset.select(range(i * split_size, (i + 1) * split_size)) for i in range(num_splits)]


def process_batch(args, batch, clm, max_target_len):
    supervised_trie = Trie()
    clm_trie = Trie() if clm else None

    for j in range(len(batch)):
        input_id, label = batch[j]["input_ids"], batch[j]["labels"]
        assert len(input_id) == len(label)
        if clm:
            flag4LossArea = False
            regionBeginIdx = -1

        max_target_len = max(max_target_len, len(label))
        for i in range(len(label) - 1):
            if label[i + 1] != -100:
                supervised_trie.insert(input_id[: i + 1], label[i + 1])
                if clm:
                    if flag4LossArea is False:
                        regionBeginIdx = i
                        flag4LossArea = True
                    if i - regionBeginIdx >= args.ngram:
                        clm_trie.insert(label[regionBeginIdx + 1 : i + 1], label[i + 1])
            elif clm and flag4LossArea:
                flag4LossArea = False

    return supervised_trie, clm_trie, max_target_len


def merge_tries(tries):
    merged_trie = Trie()
    for trie in tries:
        merged_trie.merge(trie)
    return merged_trie


def statistic(args, train_dataset, mono_dataset=None):
    supervised_trie = Trie()
    clm_trie = Trie() if args.clm else None
    max_target_len = 0

    logger.debug(f"start to make statistic")

    num_splits = os.cpu_count()
    chunks = split_dataset(train_dataset, num_splits)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_batch, args, chunk, args.clm, max_target_len) for chunk in chunks]
        for future in tqdm(futures, desc="statistic stage"):
            supervised_part, clm_part, max_len = future.result()
            supervised_trie.merge(supervised_part)
            if args.clm:
                clm_trie.merge(clm_part)
            max_target_len = max(max_target_len, max_len)

    if args.mono and args.clm:
        mono_chunks = split_dataset(mono_dataset, num_splits)
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_batch, args, chunk, args.clm, max_target_len) for chunk in mono_chunks]
            for future in tqdm(futures, desc="mono statistic stage"):
                _, clm_part, _ = future.result()
                clm_trie.merge(clm_part)

    return supervised_trie, clm_trie


def synthesis(args, train_dataset, supervised_trie, clm_trie, template):
    tokenizer = template.tokenizer
    synthesis_dict = defaultdict(list)
    cnt_list = []

    for j in tqdm(range(len(train_dataset)), desc="synthesis stage"):
        input_id, label = train_dataset[j]["input_ids"], train_dataset[j]["labels"]

        if args.clm:
            flag4LossArea = False
            regionBeginIdx = -1

        key = tuple(input_id[:-1])
        length = len(input_id)
        if synthesis_dict[key] == [] or (
            template.chat_eos_token_id not in synthesis_dict[key][-1][0]
            and template.chat_eos_token_id not in synthesis_dict[key][-2][0]
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
                        if clm_value == None or len(clm_value) == 0:
                            clm_value = supervised_value
                    assert clm_value is not None
                    synthesis_dict[key].append([supervised_value, clm_value])

                elif args.clm and flag4LossArea:
                    flag4LossArea = False

                if len(synthesis_dict) != len(cnt_list):
                    import pdb
                    pdb.set_trace()

    return synthesis_dict, cnt_list

@logger.catch
def main():
    args = parse_args()
    test(args)
    tokenizer = AutoTokenizer.from_pretrained(model_dir.get(args.model, args.model))
    template = modelType2Template[args.template](tokenizer)

    train_dataset = parse_dataset(args, template, args.dataset)
    if args.mono:
        mono_dataset = parse_dataset(args, template, args.mono_dataset)
        supervised_trie, clm_trie = statistic(args, train_dataset, mono_dataset)
    else:
        supervised_trie, clm_trie = statistic(args, train_dataset)
    synthesis_dict, cnt_list = synthesis(args, train_dataset, supervised_trie, clm_trie, template)

    logger.debug(f"length of synthesis_dict:{len(synthesis_dict)};length of cnt_list:{len(cnt_list)}")
    assert len(synthesis_dict) == len(cnt_list)

    script_path = os.path.dirname(os.path.abspath(__file__).rstrip(os.sep))
    output_base_dir = os.path.join(script_path, "train_dataset", (f"{args.template}_{args.dataset}"))
    if args.mono:
        output_base_dir += f"_mono_{args.mono_dataset.replace(',','_')}"
    if args.w_template:
        output_base_dir += "_template"

    os.makedirs(output_base_dir, exist_ok=True)

    save_chunks(synthesis_dict, chunk_size=1024, base_dir=output_base_dir, name="synthesis")
    save_chunks(cnt_list, chunk_size=1024, base_dir=output_base_dir, name="index")

    logger.debug(
        f"整合文件被保存到train_dataset/{args.template}_{args.dataset}"
        if not args.mono
        else f"{args.template}_{args.dataset}_mono_{args.mono_dataset.replace(',','_')}"
    )

def test(args):
    tokenizer = AutoTokenizer.from_pretrained(model_dir.get('gemma_2b'))
    template = modelType2Template['gemma'](tokenizer)
    mock_dataset=dname2load['test'](None)
    mono_dataset = parse_dataset(args, template, 'test')
    mock_dataset = mock_dataset.map(
        partial(
            dname2func['test'],
            template=template,
            mode=1 ,
            test=False,
        ),
        batched=True,
        num_proc=30,
        # remove_columns=train_dataset.features.keys(),
        load_from_cache_file=False,
        desc="tokenize",
    )
    supervised_trie, clm_trie = statistic(args,mock_dataset,mono_dataset)
    if args.template=='qwen2':
        if args.ngram==0:
            target_synthesis_dict={(2, 106, 1645, 108, 235285, 2182, 692, 604, 2149, 13669, 1069, 107, 108, 106, 2516, 108, 235285, 2182, 692, 604, 2149, 13669, 1069, 1): [[Counter({235285: 1}), Counter({235285: 4})], [Counter({2182: 1}), Counter({2182: 2, 798: 2})], [Counter({692: 1}), Counter({692: 2})], [Counter({604: 1}), Counter({604: 2})], [Counter({2149: 1}), Counter({2149: 2})], [Counter({13669: 1}), Counter({13669: 2})], [Counter({1069: 1}), Counter({1069: 2})], [Counter({1: 1}), Counter({1: 2})], [Counter({108: 1}), Counter({108: 2})]], (2, 106, 1645, 108, 235285, 798, 235303, 235251, 1707, 692, 107, 108, 106, 2516, 108, 235285, 798, 235303, 235251, 1707, 692, 1): [[Counter({235285: 1}), Counter({235285: 4})], [Counter({798: 1}), Counter({2182: 2, 798: 2})], [Counter({235303: 1}), Counter({235303: 2})], [Counter({235251: 1}), Counter({235251: 2})], [Counter({1707: 1}), Counter({1707: 2})], [Counter({692: 1}), Counter({692: 2})], [Counter({1: 1}), Counter({1: 2})], [Counter({108: 1}), Counter({108: 2})]]}
        elif args.ngram==4:
            target_synthesis_dict={(2, 106, 1645, 108, 235285, 2182, 692, 604, 2149, 13669, 1069, 107, 108, 106, 2516, 108, 235285, 2182, 692, 604, 2149, 13669, 1069, 1): [[Counter({235285: 1}), Counter({235285: 1})], [Counter({2182: 1}), Counter({2182: 1})], [Counter({692: 1}), Counter({692: 1})], [Counter({604: 1}), Counter({604: 1})], [Counter({2149: 1}), Counter({2149: 2})], [Counter({13669: 1}), Counter({13669: 2})], [Counter({1069: 1}), Counter({1069: 2})], [Counter({1: 1}), Counter({1: 2})], [Counter({108: 1}), Counter({108: 2})]], (2, 106, 1645, 108, 235285, 798, 235303, 235251, 1707, 692, 107, 108, 106, 2516, 108, 235285, 798, 235303, 235251, 1707, 692, 1): [[Counter({235285: 1}), Counter({235285: 1})], [Counter({798: 1}), Counter({798: 1})], [Counter({235303: 1}), Counter({235303: 1})], [Counter({235251: 1}), Counter({235251: 1})], [Counter({1707: 1}), Counter({1707: 2})], [Counter({692: 1}), Counter({692: 2})], [Counter({1: 1}), Counter({1: 2})], [Counter({108: 1}), Counter({108: 2})]]}
        elif args.ngram==1:
            target_synthesis_dict={(2, 106, 1645, 108, 235285, 2182, 692, 604, 2149, 13669, 1069, 107, 108, 106, 2516, 108, 235285, 2182, 692, 604, 2149, 13669, 1069, 1): [[Counter({235285: 1}), Counter({235285: 1})], [Counter({2182: 1}), Counter({2182: 2, 798: 2})], [Counter({692: 1}), Counter({692: 2})], [Counter({604: 1}), Counter({604: 2})], [Counter({2149: 1}), Counter({2149: 2})], [Counter({13669: 1}), Counter({13669: 2})], [Counter({1069: 1}), Counter({1069: 2})], [Counter({1: 1}), Counter({1: 2})], [Counter({108: 1}), Counter({108: 2})]], (2, 106, 1645, 108, 235285, 798, 235303, 235251, 1707, 692, 107, 108, 106, 2516, 108, 235285, 798, 235303, 235251, 1707, 692, 1): [[Counter({235285: 1}), Counter({235285: 1})], [Counter({798: 1}), Counter({2182: 2, 798: 2})], [Counter({235303: 1}), Counter({235303: 2})], [Counter({235251: 1}), Counter({235251: 2})], [Counter({1707: 1}), Counter({1707: 2})], [Counter({692: 1}), Counter({692: 2})], [Counter({1: 1}), Counter({1: 2})], [Counter({108: 1}), Counter({108: 2})]]}
    else:
        if args.ngram==1:
            target_synthesis_dict={(2, 106, 1645, 108, 235285, 2182, 692, 604, 2149, 13669, 1069, 107, 108, 106, 2516, 108, 235285, 2182, 692, 604, 2149, 13669, 1069, 1): [[Counter({235285: 1}), Counter({235285: 1})], [Counter({2182: 1}), Counter({2182: 2, 798: 2})], [Counter({692: 1}), Counter({692: 2})], [Counter({604: 1}), Counter({604: 2})], [Counter({2149: 1}), Counter({2149: 2})], [Counter({13669: 1}), Counter({13669: 2})], [Counter({1069: 1}), Counter({1069: 2})], [Counter({1: 1}), Counter({1: 2})], [Counter({108: 1}), Counter({108: 2})]], (2, 106, 1645, 108, 235285, 798, 235303, 235251, 1707, 692, 107, 108, 106, 2516, 108, 235285, 798, 235303, 235251, 1707, 692, 1): [[Counter({235285: 1}), Counter({235285: 1})], [Counter({798: 1}), Counter({2182: 2, 798: 2})], [Counter({235303: 1}), Counter({235303: 2})], [Counter({235251: 1}), Counter({235251: 2})], [Counter({1707: 1}), Counter({1707: 2})], [Counter({692: 1}), Counter({692: 2})], [Counter({1: 1}), Counter({1: 2})], [Counter({108: 1}), Counter({108: 2})]]}
        elif args.ngram==4:
            target_synthesis_dict={(2, 106, 1645, 108, 235285, 2182, 692, 604, 2149, 13669, 1069, 107, 108, 106, 2516, 108, 235285, 2182, 692, 604, 2149, 13669, 1069, 1): [[Counter({235285: 1}), Counter({235285: 1})], [Counter({2182: 1}), Counter({2182: 1, 798: 1})], [Counter({692: 1}), Counter({692: 1})], [Counter({604: 1}), Counter({604: 1})], [Counter({2149: 1}), Counter({2149: 1})], [Counter({13669: 1}), Counter({13669: 1})], [Counter({1069: 1}), Counter({1069: 1})], [Counter({1: 1}), Counter({1: 1})], [Counter({108: 1}), Counter({108: 1})]], (2, 106, 1645, 108, 235285, 798, 235303, 235251, 1707, 692, 107, 108, 106, 2516, 108, 235285, 798, 235303, 235251, 1707, 692, 1): [[Counter({235285: 1}), Counter({235285: 1})], [Counter({798: 1}), Counter({2182: 1, 798: 1})], [Counter({235303: 1}), Counter({235303: 1})], [Counter({235251: 1}), Counter({235251: 1})], [Counter({1707: 1}), Counter({1707: 1})], [Counter({692: 1}), Counter({692: 1})], [Counter({1: 1}), Counter({1: 1})], [Counter({108: 1}), Counter({108: 1})]]}
    target_cnt_list=[[(15, 24)], [(14, 22)]]
    synthesis_dict, cnt_list = synthesis(args,mock_dataset,supervised_trie, clm_trie,template)
    if target_synthesis_dict!=synthesis_dict:
        import pdb
        pdb.set_trace() 
    assert target_synthesis_dict==synthesis_dict 
    assert cnt_list == target_cnt_list
if __name__ == "__main__":
    main()
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
from transformers import (
    AutoTokenizer,
    AutoConfig,
    DataCollatorForSeq2Seq,
    AutoModelForCausalLM,
)
import os
from ..eval.post_process import dname2post
from torch.utils.data import DataLoader
from ..eval.load_func import dname2load
from ..eval.samplingparam import dname2samplingparams
from vllm import LLM
from ..config import *
from ..dataset import SpecialDataset, SpecialDataCollator
import torch.nn.functional as F
import time


def parse_args():
    parser = ArgumentParser()

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
        "--mix_ratio", default=0.8, type=ast.literal_eval, help="sft信号的融合比例"
    )
    parser.add_argument("--w_template", default=True, type=ast.literal_eval)
    parser.add_argument("--template", type=str)
    return parser.parse_args()


def load_msgpack_file(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)  # ,strict_map_key=False,strict_types =True


def count_top_p_elements(tensor):
    # 计算累积概率
    sorted_tensor, _ = torch.sort(tensor, descending=True)
    cumulative_probabilities = torch.cumsum(sorted_tensor, dim=1)

    # 初始化结果字典
    top_p_counts = {p: 0 for p in [0.1 * i for i in range(1, 10)]}
    # import pdb

    # pdb.set_trace()
    # 查找满足累积概率大于等于每个p阈值的第一个索引
    for p in top_p_counts.keys():
        for i in range(tensor.size(0)):
            count = torch.sum(cumulative_probabilities[i] < p).item() + 1
            top_p_counts[p] += count

    return top_p_counts


def merge_dicts(dict1, dict2):
    for key in dict2:
        dict1[key] += dict2[key]
    return dict1


def convert_onehot_to_smoothed(onehot_labels, epsilon=0.1):
    """
    将独热编码转换为label smoothing分布

    Args:
        onehot_labels (torch.Tensor): 独热编码标签 shape=(batch_size, num_classes)
        epsilon (float): 平滑因子,默认0.1

    Returns:
        torch.Tensor: 平滑后的标签分布
    """
    num_classes = onehot_labels.size(-1)
    smoothed_labels = onehot_labels * (1 - epsilon) + epsilon / num_classes
    return smoothed_labels


@logger.catch
def main():
    args = parse_args()

    model_list = args.model.split(",")

    # os.makedirs(args.output_path,exist_ok=True)
    script_path = os.path.dirname(os.path.abspath(__file__))
    print("script_path", script_path)

    import time

    for m in model_list:
        m = model_dir.get(m)
        tokenizer = AutoTokenizer.from_pretrained(m)
        tokenizer.padding_side = "left"
        template = modelType2Template[args.template](tokenizer)
        model_name = os.path.basename(
            m.rstrip(os.sep)
        )  # 不去掉sep，碰到 a/b/ 就会读到空。
        record_list = []

        @logger.catch
        def load_dataset():

            data_folder_path = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__).rstrip(os.sep)).rstrip(os.sep)
            )

            def find_msgpack_chunk_files(
                base_dir,
                name,
            ):
                """查找与基准文件名匹配的所有 msgpack 分块文件。"""

                chunk_files = [
                    os.path.join(base_dir, f)
                    for f in os.listdir(base_dir)
                    if f.startswith(name)
                    and (
                        f.endswith("part0.msgpack")
                        or f.endswith("part1.msgpack")
                        or f.endswith("part3.msgpack")
                        or f.endswith("part206.msgpack")
                        or f.endswith("part207.msgpack")
                        or f.endswith("part130.msgpack")
                        or f.endswith("part131.msgpack")
                        or f.endswith("part132.msgpack")
                        or f.endswith("part133.msgpack")
                        or f.endswith("part134.msgpack")
                    )
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

            base_dir = (
                f"{data_folder_path}/train_dataset/{args.template}_{args.dataset}"
            )
            if args.w_template:
                base_dir += "_template"
            synthesis = load_msgpack_chunks(
                find_msgpack_chunk_files(
                    base_dir,
                    name="synthesis",
                )
            )
            index = load_msgpack_chunks(
                find_msgpack_chunk_files(
                    base_dir,
                    name="index",
                )
            )

            train_dataset = SpecialDataset(
                synthesis,
                index,
                embedding_size=-1,  # 遗留行为，dataset用不上这个参数
                zero_prob=args.zero_prob,
                div_mode=args.div_mode,
            )
            return train_dataset

        print("开始加载数据\n")
        start_time = time.time()
        train_dataset = load_dataset()

        end_time = time.time()
        print("数据集长度是=", len(train_dataset))
        print("数据加载时间为=", end_time - start_time, "\n")
        with torch.inference_mode():
            model = AutoModelForCausalLM.from_pretrained(
                m,
                attn_implementation="flash_attention_2",
                torch_dtype="bfloat16",
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(m, padding_side="left")

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            collator = SpecialDataCollator(
                tokenizer=tokenizer,
                zero_prob=args.zero_prob,
                embedding_size=model.lm_head.weight.size()[0],
                div_mode=args.div_mode,
                mix=False,
                mix_ratio=args.mix_ratio,
                pt=False,
                offline=False,
            )
            dataloader = DataLoader(
                dataset=train_dataset,
                batch_size=2,
                collate_fn=collator,
                num_workers=16,
                pin_memory=True,
            )
            (
                supervised_similarity,
                clm_similarity,
                naive_label_similarity,
                mix_similarity,
                var,
            ) = (
                0,
                0,
                0,
                0,
                0,
            )
            from collections import defaultdict

            cumulative_top_p_counts = defaultdict(int)
            for d in tqdm(dataloader):

                response = model(
                    input_ids=d["input_ids"],
                    attention_mask=d["attention_mask"],
                ).logits

                temp_last_logits = torch.cat(
                    [
                        row[start:end]
                        for row, turn in zip(response, d["valid_label_index_list"])
                        for start, end in turn
                    ]
                )

                last_logits = torch.nn.functional.softmax(
                    temp_last_logits,
                    dim=-1,
                )
                temp_topp_dict = count_top_p_elements(last_logits)
                cumulative_top_p_counts = merge_dicts(
                    cumulative_top_p_counts, temp_topp_dict
                )
                var += torch.sum(torch.var(temp_last_logits, dim=-1)).item()
                real_label = torch.cat(
                    [
                        torch.cat(
                            (
                                row[start + 1 : end],
                                torch.tensor([tokenizer.eos_token_id]),
                            )
                        )
                        for row, turn in zip(
                            d["input_ids"], d["valid_label_index_list"]
                        )
                        for start, end in turn
                    ]
                )
                label_tensor = F.one_hot(real_label, model.lm_head.weight.size()[0]).to(
                    last_logits.device
                )

                label_tensor = convert_onehot_to_smoothed(label_tensor)
                temp_label_similarity = F.cosine_similarity(last_logits, label_tensor)

                naive_label_similarity += torch.sum(temp_label_similarity).item()

                c = time.time()
                # print('forward',b-a,'post',c-b)
            print(
                supervised_similarity,
                clm_similarity,
                naive_label_similarity,
                mix_similarity,
                var,
            )
            print(cumulative_top_p_counts)
            supervised_ratio_naive_label_similarity = (
                supervised_similarity / naive_label_similarity
            )
            clm_ratio_naive_label_similarity = clm_similarity / naive_label_similarity
            mix_ratio_naive_label_similarity = mix_similarity / naive_label_similarity

            config_str = f"{args.model}---dm_{args.div_mode}---zp_{args.zero_prob}---mr_{args.mix_ratio}"
            result = {
                "supervised_similarity": supervised_similarity,
                "clm_similarity": clm_similarity,
                "naive_label_similarity": naive_label_similarity,
                "mix_similarity": mix_similarity,
                "supervised_similarity-ratio-naive_label_similarity": supervised_ratio_naive_label_similarity,
                "clm_similarity-ratio-naive_label_similarity": clm_ratio_naive_label_similarity,
                "mix_similarity-ratio-naive_label_similarity": mix_ratio_naive_label_similarity,
            }
            result.update(cumulative_top_p_counts)
            print("\nConfiguration:", config_str, "\n")
            print("\nResults:", result)
            # 保存到文件中
            output_path = args.output_path
            if os.path.exists(output_path):
                with open(output_path, "r") as f:
                    results = json.load(f)
            else:
                results = {}

            # 以配置字符串为键，结果为值保存
            results[config_str] = result

            with open(output_path, "w") as f:
                json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()

from collections import Counter
from functools import partial
from torch.utils.data import Dataset
import torch
from scipy.optimize import root_scalar, fsolve
import faulthandler

# 在import之后直接添加以下启用代码即可
faulthandler.enable()


def transform_to_log_prob(
    knns,
    vocab_size=None,
    zero_prob=None,
):

    if len(knns) == 0:
        # return torch.zeros((vocab_size))
        return None
    else:

        # 预先计算,免得在多次fun的迭代里都要重算
        zero_count_per_tensor = torch.sum(knns == 0, dim=-1)
        # 分母
        bsz = knns.size(0)

        def fun(x):
            if x <= 0:
                return 10000

            # tensor_with_temperature = knns / x
            # exp_tensor = torch.exp(tensor_with_temperature)
            # sum_exp = torch.sum(exp_tensor, dim=-1)
            # result = torch.sum(zero_count_per_tensor / sum_exp) - bsz * zero_prob

            x = knns / x
            x = torch.exp(x)
            x = torch.sum(x, dim=-1)
            result = torch.sum(zero_count_per_tensor / x) - bsz * zero_prob

            return result.item()

        # 区间大1-100的时候很适合Ridder，区间小1-10/1-50的时候toms748更好
        result = root_scalar(fun, bracket=[0.01, 50], method="toms748")
        knn_temperature = result.root
    probs = torch.nn.functional.softmax(knns / knn_temperature, dim=-1)

    return probs


def frequency(x, xmax=50):
    return (x / xmax) ** 0.75 if x < xmax else 1


def optimized_stack(supervised, embedding_size):
    """
    Optimizes the provided code for stacking Counters into a PyTorch tensor.

    Args:
        supervised: A list of Counter objects.
        embedding_size: The desired size of the embedding dimension.

    Returns:
        A PyTorch tensor representing the stacked Counters.
    """

    # Pre-allocate the tensor for efficiency
    x = torch.zeros(len(supervised), embedding_size)

    # Iterate through the Counters and directly update the tensor
    for i, counter in enumerate(supervised):
        for key, value in counter.items():
            if key < embedding_size:  # Handle out-of-bounds indices
                x[i, key] = value

    return x


# def directly_softmax(supervised, embedding_size):

#     x = torch.zeros(len(supervised), embedding_size)

#     # Iterate through the Counters and directly update the tensor
#     for i, counter in enumerate(supervised):

#         temp_values=torch.nn.functional.softmax(torch.tensor(list(counter.values()),dtype=float),dim=-1)
#         # if len(counter.values())>1:
#         #     import pdb
#         #     pdb.set_trace()
#         for j,key in enumerate(counter.keys()):
#             x[i][key]=temp_values[j]
#         # print(x[0],x[1],temp_values)
#     # import pdb
#     # pdb.set_trace()
#     return x


def find_temperature(x, target_index, zero_prob=0.05, tol=1e-6, max_iter=100):
    low, high = 1e-7, 100  # 初始搜索范围
    for _ in range(max_iter):
        T = (low + high) / 2.0
        softmax_x = torch.nn.functional.softmax(x / T, dim=-1)
        if abs(softmax_x[target_index] - zero_prob) < tol:
            return T
        if softmax_x[target_index] < zero_prob:
            low = T
        else:
            high = T
    return T


# 6月28日下午的优化版本
def directly_softmax(supervised, embedding_size, div=False, zero_prob=0):
    x = torch.zeros(len(supervised), embedding_size)

    for i, counter in enumerate(supervised):
        # Convert counter values to a tensor and compute softmax
        indices = torch.tensor(list(counter.keys()), dtype=torch.long)
        if zero_prob == 0:
            values = torch.tensor(list(counter.values()), dtype=torch.float32)
            if div:
                temp_values = torch.div(values, torch.sum(values))
            else:
                temp_values = torch.nn.functional.softmax(values, dim=-1)
            x[i].scatter_(0, indices, temp_values)
        else:
            values = torch.tensor(list(counter.values()) + [0], dtype=torch.float32)
            if div:
                pass
            else:
                T = find_temperature(values, target_index=-1, zero_prob=zero_prob)
                temp_result = torch.nn.functional.softmax(values / T, dim=-1)
                temp_values, zero_values = temp_result[:-1], temp_result[-1].item()
                zero_values /= embedding_size - len(counter.values())
                x[i] = torch.full_like(x[i], zero_values)
                x[i][indices] = temp_values
    return x


import torch
from collections import Counter


class SpecialDataset(Dataset):
    def __init__(
        self,
        synthesis_dict: dict[str : tuple[list[Counter], list[Counter]]],
        cnt_list: list[list[tuple]],  # 在单轮对话里每个list里应该只有tuple
        embedding_size,
        zero_prob,
        div_mode=False,
        pt=False,
    ):
        self.pt = pt
        synthesis_dict = [data_sample for data_sample in synthesis_dict.items()]
        self.input_ids = [
            list(synthesis_dict[i][0]) for i in range(len(synthesis_dict))
        ]
        if not pt:

            self.supervised = [
                [synthesis_dict[i][1][j][0] for j in range(len(synthesis_dict[i][1]))]
                for i in range(len(synthesis_dict))
            ]
            # self.supervised = [synthesis_dict[i][1][j][0]  for i in range(len(synthesis_dict))  for j in range(len(synthesis_dict[i]))]
            self.clm = [
                [synthesis_dict[i][1][j][1] for j in range(len(synthesis_dict[i][1]))]
                for i in range(len(synthesis_dict))
            ]

        else:
            self.clm = [
                [synthesis_dict[i][1][j][0] for j in range(len(synthesis_dict[i][1]))]
                for i in range(len(synthesis_dict))
            ]
        self.valid_label_index_list = cnt_list

        self.embedding_size = embedding_size
        self.div_mode = div_mode
        self.zero_prob = zero_prob

    def __getitem__(self, index):

        if not self.pt:
            # 很不耗时，e-6
            return {
                "input_ids": self.input_ids[index],
                "supervised": self.supervised[index],
                "clm": self.clm[index],
                "embedding_size": self.embedding_size,
                "div_mode": self.div_mode,
                "valid_label_index_list": self.valid_label_index_list[index],
            }
        else:
            return {
                "input_ids": self.input_ids[index],
                "clm": self.clm[index],
                "embedding_size": self.embedding_size,
                "div_mode": self.div_mode,
                "valid_label_index_list": self.valid_label_index_list[index],
            }

    def __len__(self):
        return len(self.input_ids)


class OfflineDataset(Dataset):
    def __init__(
        self,
        synthesis_list,
    ):

        self.synthesis_list = synthesis_list

    def __getitem__(self, index):

        return self.synthesis_list[index]

    def __len__(self):
        return len(self.synthesis_list)


class SpecialDataCollator:
    def __init__(
        self,
        tokenizer,
        zero_prob,
        embedding_size,
        div_mode,
        mix,
        mix_ratio,
        pt,
        offline,
    ) -> None:
        self.tokenizer = tokenizer
        self.zero_prob = zero_prob
        self.embedding_size = embedding_size
        self.div_mode = div_mode
        self.mix = mix
        self.mix_ratio = mix_ratio
        self.pt = pt
        self.offline = offline  # this flag only useful for data loading, don't set it to true while offline process

    def __call__(self, batch) -> torch.Any:

        input_ids = [d["input_ids"] for d in batch]
        input_ids_len = list(len(input_id) for input_id in input_ids)
        # input_ids_max_len = max(input_ids_len)
        input_ids = self.tokenizer.pad(
            {"input_ids": input_ids}, return_tensors="pt", padding=True
        )
        input_ids_max_len = input_ids["input_ids"].shape[-1]

        # temp_for_debug = [i["valid_label_index_list"] for i in batch]
        # print(input_ids.input_ids.shape)
        # print(temp_for_debug)
        # print(input_ids_max_len, input_ids_len)
        valid_label_index_list = []
        # 这个东西很复杂……，因为pad之后前面会变长，所以前面还要去掉pad的位置。
        # 不要就地改动原batch的内容，不然会与auto_find_bsz冲突。

        # 不耗时。e-5
        for i, d in enumerate(batch):
            length_diff = input_ids_max_len - input_ids_len[i]
            temp_index_list = []
            for j in range(len(d["valid_label_index_list"])):
                temp_index_list.append(
                    (
                        length_diff + d["valid_label_index_list"][j][0],
                        length_diff + d["valid_label_index_list"][j][1],
                    )
                )
            valid_label_index_list.append(temp_index_list)

  

        if self.offline:

            if self.mix:
                all_prob_mix = torch.cat([d["all_prob_mix"] for d in batch], dim=0)
                mix_cnt = torch.cat([d["mix_cnt"] for d in batch], dim=0)
                return {
                    "input_ids": input_ids.input_ids,
                    "attention_mask": input_ids.attention_mask,
                    "all_prob_mix": all_prob_supervised,
                    "valid_label_index_list": valid_label_index_list,
                    "mix_cnt": supervised_cnt,
                }
            else:
                all_prob_clm = torch.cat([d["all_prob_clm"] for d in batch], dim=0)
                clm_cnt = torch.cat([d["clm_cnt"] for d in batch], dim=0)
                if not self.pt:
                    all_prob_supervised = torch.cat(
                        [d["all_prob_supervised"] for d in batch], dim=0
                    )
                    supervised_cnt = torch.cat(
                        [d["supervised_cnt"] for d in batch], dim=0
                    )
                    return {
                        "input_ids": input_ids.input_ids,
                        "attention_mask": input_ids.attention_mask,
                        "all_prob_supervised": all_prob_supervised,
                        "all_prob_clm": all_prob_clm,
                        "valid_label_index_list": valid_label_index_list,
                        "supervised_cnt": supervised_cnt,
                        "clm_cnt": clm_cnt,
                    }

                return {
                    "input_ids": input_ids.input_ids,
                    "attention_mask": input_ids.attention_mask,
                    "all_prob_clm": all_prob_clm,
                    "valid_label_index_list": valid_label_index_list,
                    "clm_cnt": clm_cnt,
                }

        # TIME --------------------------------------------------------------
        # e-4 不算耗时
        if not self.pt:
            # 相较于input_ids，我们解开了对每个元素的list包围，使得一个batch的target从bsz，seqlen坍缩成了bsz x seqlen
            supervised = [item for d in batch for item in d["supervised"]]
        # if not all(len(counter) == 1 for counter in supervised):
        #     import pdb
        #     pdb.set_trace()
        # else:
        #     return {}
        clm = [item for d in batch for item in d["clm"]]
        # ----------------------------------------------------------------

        if self.div_mode:
            if not self.pt:
                x_sup = optimized_stack(supervised, self.embedding_size)
            x_clm = optimized_stack(clm, self.embedding_size)
            if self.zero_prob == 0:
                if not self.pt:
                    all_prob_supervised = directly_softmax(
                        supervised, self.embedding_size, div=True
                    )
                all_prob_clm = directly_softmax(clm, self.embedding_size, div=True)

            else:
                if not self.pt:
                    all_prob_supervised = (
                        (1 - self.zero_prob)
                        * x_sup
                        / torch.sum(x_sup, dim=-1, keepdim=True)
                    )
                    non_zero_cnt = torch.sum(x_sup != 0, keepdim=True, dim=-1)
                    temp_zero_prob = self.zero_prob / (
                        self.embedding_size - non_zero_cnt
                    )
                    all_prob_supervised = torch.where(
                        all_prob_supervised == 0, temp_zero_prob, all_prob_supervised
                    )

                all_prob_clm = (
                    (1 - self.zero_prob)
                    * x_clm
                    / torch.sum(x_clm, dim=-1, keepdim=True)
                )
                zero_cnt = torch.sum(x_clm != 0, keepdim=True, dim=-1)
                temp_zero_prob = self.zero_prob / (self.embedding_size - zero_cnt)
                all_prob_clm = torch.where(
                    all_prob_clm == 0, temp_zero_prob, all_prob_clm
                )
        else:

            if self.zero_prob == 0:
                if not self.pt:
                    all_prob_supervised = directly_softmax(
                        supervised, self.embedding_size, zero_prob=self.zero_prob
                    )
                all_prob_clm = directly_softmax(
                    clm, self.embedding_size, zero_prob=self.zero_prob
                )

            else:
                # import time

                # aaa = time.time()
                # 余留行为
                if not self.pt:
                    all_prob_supervised = directly_softmax(
                        supervised, self.embedding_size, zero_prob=self.zero_prob
                    )
                all_prob_clm = directly_softmax(
                    clm, self.embedding_size, zero_prob=self.zero_prob
                )

                # 下面是数值解，batch模式
                # bbb = time.time()
                # x_sup = optimized_stack(supervised, self.embedding_size)
                # x_clm = optimized_stack(clm, self.embedding_size)
                # all_prob_supervised1 = transform_to_log_prob(
                #     x_sup, zero_prob=self.zero_prob
                # )
                # all_prob_clm = transform_to_log_prob(x_clm, zero_prob=self.zero_prob)
                # print(time.time() - bbb, bbb - aaa)
                # print(torch.allclose(all_prob_supervised, all_prob_supervised1))
                # import pdb
                # pdb.set_trace()
        if not self.pt:
            supervised_cnt = torch.tensor(
                [frequency(sum(xx.values()), xmax=10) for xx in supervised]
            )
        clm_cnt = torch.tensor([frequency(sum(xx.values())) for xx in clm])

        if self.mix:
            if self.pt:
                logger.debug("不允许结合mix和预训练")
                exit()
            all_prob_supervised = (
                self.mix_ratio * all_prob_supervised
                + (1 - self.mix_ratio) * all_prob_clm
            )
            # supervised_cnt=self.mix_ratio*supervised_cnt+(1-self.mix_ratio)*clm_cnt 627日晚上注释
            supervised_cnt = supervised_cnt + clm_cnt

            return {
                "input_ids": input_ids.input_ids,
                "attention_mask": input_ids.attention_mask,
                "all_prob_mix": all_prob_supervised,
                "valid_label_index_list": valid_label_index_list,
                "mix_cnt": supervised_cnt,
            }
        if not self.pt:
            return {
                "input_ids": input_ids.input_ids,
                "attention_mask": input_ids.attention_mask,
                "all_prob_supervised": all_prob_supervised,
                "all_prob_clm": all_prob_clm,
                "valid_label_index_list": valid_label_index_list,
                "supervised_cnt": supervised_cnt,
                "clm_cnt": clm_cnt,
            }
        else:
            return {
                "input_ids": input_ids.input_ids,
                "attention_mask": input_ids.attention_mask,
                "all_prob_clm": all_prob_clm,
                "valid_label_index_list": valid_label_index_list,
                "clm_cnt": clm_cnt,
            }


if __name__ == "__main__":
    # torch.multiprocessing.set_sharing_strategy('file_descriptor')
    import ast
    import json
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
    )
    from torch.utils.data import Dataset, DataLoader
    import datasets
    from special_trainer import KLTrainer
    import pickle
    from config import model_dir, dataset_dir
    import torch
    from argparse import ArgumentParser
    from loguru import logger
    import warnings
    import os

    def parse_args():
        parser = ArgumentParser()
        parser.add_argument("--model", default="gemma_2b")
        parser.add_argument("--dataset", default="alpaca_gpt4")
        parser.add_argument("--div_mode", default=True, type=ast.literal_eval)
        parser.add_argument("--output_dir")
        parser.add_argument(
            "--fa2", action="store_true", help="decide to use fa2 or not"
        )
        parser.add_argument(
            "--lora", action="store_true", help="decide to use lora or not"
        )
        parser.add_argument("--zero_prob", default=0, type=ast.literal_eval)
        parser.add_argument("--gradient_accumulation_steps", default=8, type=int)
        parser.add_argument("--total_bsz", default=64, type=int)
        parser.add_argument(
            "--weighted",
            action="store_true",
            help="decide to use token level freq weight",
        )
        parser.add_argument(
            "--mix",
            default=False,
            type=ast.literal_eval,
            help="decide to use token level freq weight",
        )
        parser.add_argument(
            "--mix_ratio", default=0.8, type=ast.literal_eval, help="sft信号的融合比例"
        )
        return parser.parse_args()

    args = parse_args()
    model_dir = model_dir.get(args.model, args.model)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        # device_map="auto",  # 在显存不够的时候优先考虑流水线并行吧。 这样不需要考虑变化的总bsz
        attn_implementation="flash_attention_2" if args.fa2 else "sdpa",
    )
    model_type = model.config.model_type
    embedding_size = model.lm_head.weight.size()[
        0
    ]  # 取lm_head比较安全，因为有些模型embedding layer会取不同的名字

    collator = SpecialDataCollator(
        tokenizer,
        zero_prob=args.zero_prob,
        embedding_size=embedding_size,
        div_mode=args.div_mode,
        mix=args.mix,
        mix_ratio=args.mix_ratio,
    )

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

    @logger.catch
    def load_dataset():
        script_path = os.path.dirname(os.path.abspath(__file__).rstrip(os.sep))
        # with open(
        #     f"{script_path}/train_dataset/{model_type}_{args.dataset}_synthesis.pkl", "rb"
        # ) as f:
        #     synthesis = pickle.load(f)

        # with open(
        #     f"{script_path}/train_dataset/{model_type}_{args.dataset}_index.pkl", "rb"
        # ) as f:
        #     index = pickle.load(f)

        base_dir = f"{script_path}/train_dataset/{model_type}_{args.dataset}"

        synthesis = load_msgpack_chunks(
            find_msgpack_chunk_files(base_dir, name="synthesis")
        )
        index = load_msgpack_chunks(find_msgpack_chunk_files(base_dir, name="index"))

        train_dataset = SpecialDataset(
            synthesis,
            index,
            embedding_size,
            zero_prob=args.zero_prob,
            div_mode=args.div_mode,
        )
        return train_dataset

    train_dataset = load_dataset()
    dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        collate_fn=collator,
        num_workers=8,
        pin_memory=True,
    )

    from tqdm import tqdm

    # from sklearn import UMAP
    # all_prob_supervised=
    # all_prob_clm=[]
    for d in tqdm(dataloader):
        # import pdb

        # pdb.set_trace()
        # print(d['all_prob_supervised'].shape)
        # all_prob_supervised.append()
        continue

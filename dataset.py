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
        # 不需要拟合温度的情况。
        if zero_prob == 0:
            knn_temperature = 1e-6  # 要放大，才能压低概率
        else:

            # 预先计算,免得在多次fun的迭代里都要重算
            zero_count_per_tensor = torch.sum(knns == 0, dim=-1)
            # 分母
            bsz = knns.size(0)

            def fun(x):
                if x <= 0:
                    return 10000

                tensor_with_temperature = knns / x
                exp_tensor = torch.exp(tensor_with_temperature)
                sum_exp = torch.sum(exp_tensor, dim=-1)
                result = torch.sum(zero_count_per_tensor / sum_exp) - bsz * zero_prob

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
    ):

        synthesis_dict = [data_sample for data_sample in synthesis_dict.items()]
        self.supervised = [
            [synthesis_dict[i][1][j][0] for j in range(len(synthesis_dict[i][1]))]
            for i in range(len(synthesis_dict))
        ]
        # self.supervised = [synthesis_dict[i][1][j][0]  for i in range(len(synthesis_dict))  for j in range(len(synthesis_dict[i]))]
        self.clm = [
            [synthesis_dict[i][1][j][1] for j in range(len(synthesis_dict[i][1]))]
            for i in range(len(synthesis_dict))
        ]
        self.input_ids = [list(synthesis_dict[i][0]) for i in range(len(synthesis_dict))]

        self.valid_label_index_list = cnt_list

        self.embedding_size = embedding_size
        self.div_mode = div_mode
        self.zero_prob = zero_prob

    def __getitem__(self, index):

        # 很不耗时，e-6
        return {
            "input_ids": self.input_ids[index],
            "supervised": self.supervised[index],
            "clm": self.clm[index],
            "embedding_size": self.embedding_size,
            "div_mode": self.div_mode,
            "valid_label_index_list": self.valid_label_index_list[index],
        }

    def __len__(self):
        return len(self.input_ids)


from torch.nn.utils.rnn import pad_sequence


class SpecialDataCollator:
    def __init__(self, tokenizer, zero_prob, embedding_size, div_mode) -> None:
        self.tokenizer = tokenizer
        self.zero_prob = zero_prob
        self.embedding_size = embedding_size
        self.div_mode = div_mode

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
        # 千万不要改动原batch的内容，不然会与auto_find_bsz冲突。

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

        # if valid_label_index_list==[[[711, 719]], [[503, 675]], [[20, 375]], [[551, 577]], [[486, 611]], [[117, 472]], [[626, 740]], [[831, 847]]]:
        #     import pdb
        #     pdb.set_trace()
        # print(valid_label_index_list)
        # print("-----------------------------")

        # all_prob_supervised = [d["all_prob_supervised"] for d in batch]
        # all_prob_clm = [d["all_prob_clm"] for d in batch]
        # supervised_cnt = [d["supervised_cnt"] for d in batch]
        # clm_cnt = [d["clm_cnt"] for d in batch]

        # TIME --------------------------------------------------------------
        # e-4 不算耗时
        # 相较于input_ids，我们解开了对每个元素的list包围，使得一个batch的target从bsz，seqlen坍缩成了bsz x seqlen
        supervised = [item for d in batch for item in d["supervised"]]
        clm = [item for d in batch for item in d["clm"]]
        # ----------------------------------------------------------------

        x_sup = optimized_stack(supervised, self.embedding_size)
        x_clm = optimized_stack(clm, self.embedding_size)

        if self.div_mode:

            if self.zero_prob == 0:
                all_prob_supervised = x_sup / torch.sum(x_sup, dim=-1, keepdim=True)
            else:
                all_prob_supervised = (
                    (1 - self.zero_prob)
                    * x_sup
                    / torch.sum(x_sup, dim=-1, keepdim=True)
                )
                non_zero_cnt = torch.sum(x_sup != 0, keepdim=True, dim=-1)
                temp_zero_prob = self.zero_prob / (self.embedding_size - non_zero_cnt)
                all_prob_supervised = torch.where(
                    all_prob_supervised == 0, temp_zero_prob, all_prob_supervised
                )

            if self.zero_prob == 0:
                all_prob_clm = x_clm / torch.sum(x_clm, dim=-1, keepdim=True)
            else:
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
            all_prob_supervised = transform_to_log_prob(x_sup, zero_prob=self.zero_prob)
            all_prob_clm = transform_to_log_prob(x_clm, zero_prob=self.zero_prob)

        supervised_cnt = torch.tensor(
            [frequency(sum(xx.values()), xmax=10) for xx in supervised]
        )
        clm_cnt = torch.tensor([frequency(sum(xx.values())) for xx in clm])

        return {
            "input_ids": input_ids.input_ids,
            "attention_mask": input_ids.attention_mask,
            "all_prob_supervised": all_prob_supervised,
            "all_prob_clm": all_prob_clm,
            "valid_label_index_list": valid_label_index_list,
            "supervised_cnt": supervised_cnt,
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
    )

    @logger.catch
    def load_dataset():
        script_path = os.path.dirname(os.path.abspath(__file__).rstrip(os.sep))
        with open(
            f"{script_path}/train_dataset/{model_type}_{args.dataset}_synthesis.pkl",
            "rb",
        ) as f:
            synthesis = pickle.load(f)

        with open(
            f"{script_path}/train_dataset/{model_type}_{args.dataset}_index.pkl", "rb"
        ) as f:
            index = pickle.load(f)

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
    from sklearn import UMAP
    # all_prob_supervised=
    # all_prob_clm=[]
    for d in tqdm(dataloader):
        # print(d['all_prob_supervised'].shape)
        # all_prob_supervised.append()
        continue

from collections import Counter
from functools import partial
from torch.utils.data import Dataset
import torch
from scipy.optimize import  root_scalar
from memory_profiler import profile


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
            knn_temperature = 0.000001  # 要放大，才能压低概率
        else:

            # 预先计算,免得在多次fun的迭代里都要重算
            zero_count_per_tensor = torch.sum(knns == 0, dim=-1)
            non_zero_index = torch.nonzero(knns)
            # 分母
            bsz = knns.size(0)

            def fun(knns,x):
                if x <= 0:
                    return 10000

                tensor_with_temperature = knns / x

                exp_tensor = torch.exp(tensor_with_temperature)

                sum_exp = torch.sum(exp_tensor, dim=-1)
                result = torch.sum(zero_count_per_tensor / sum_exp) - bsz * zero_prob
                return result.item()

            # 区间大1-100的时候很适合Ridder，区间小1-10/1-50的时候toms748更好
            result = root_scalar(partial(fun,knns=knns), bracket=[0.01, 100], method="Ridder")
            knn_temperature = result.root

        probs = torch.nn.functional.softmax(knns / knn_temperature, dim=-1)

    return probs


def frequency(x, xmax=50):
    return (x / xmax) ** 0.75 if x < xmax else 1


def optimized_stack_batch(supervised, embedding_size):
    """
    Optimizes the provided code for stacking Counters into a PyTorch tensor.

    Args:
        supervised: A list of Counter objects.
        embedding_size: The desired size of the embedding dimension.

    Returns:
        A PyTorch tensor representing the stacked Counters.
    """

    # Pre-allocate the tensor for efficiency
    import pdb
    pdb.set_trace()
    x = torch.stack(
            [
                torch.bincount(
                    torch.tensor(list(xx.elements())), minlength=embedding_size
                )
                for xx in supervised
            ]
        )
    x = torch.zeros(len(supervised), embedding_size) # 因为每个位置很难超过int16

    # Iterate through the Counters and directly update the tensor
    for i, counter in enumerate(supervised):
        for key, value in counter.items():
            if key < embedding_size:  # Handle out-of-bounds indices
                x[i, key] = value

    return x

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
    x = torch.zeros(len(supervised), embedding_size) # 因为每个位置很难超过int16

    # Iterate through the Counters and directly update the tensor
    for i, counter in enumerate(supervised):
        for key, value in counter.items():
            if key < embedding_size:  # Handle out-of-bounds indices
                x[i, key] = value

    return x


def get_data(
    supervised,
    clm,
    input_ids,
    valid_label_index_list,
    embdding_size,
    zero_prob,
    div_mode,
):

    temp_dict = {}

    
    # supervised_cnt = [frequency(sum(xx.values()), xmax=10) for xx in supervised]
    # clm_cnt = [frequency(sum(xx.values())) for xx in clm]
    supervised_cnt = None
    clm_cnt =None
    if div_mode:

        x = optimized_stack(supervised, embdding_size)

        all_prob_supervised = (1 - zero_prob) * x / torch.sum(x, dim=-1, keepdim=True)
        zero_cnt = torch.sum(x != 0, keepdim=True, dim=-1)
        temp_zero_prob = zero_prob / (embdding_size - zero_cnt)
        all_prob_supervised = torch.where(
            all_prob_supervised == 0, temp_zero_prob, all_prob_supervised
        )

        x = optimized_stack(clm, embdding_size)

        all_prob_clm = (1 - zero_prob) * x / torch.sum(x, dim=-1, keepdim=True)
        zero_cnt = torch.sum(x != 0, keepdim=True, dim=-1)
        temp_zero_prob = zero_prob / (embdding_size - zero_cnt)
        all_prob_clm = torch.where(all_prob_clm == 0, temp_zero_prob, all_prob_clm)

    else:
        x = optimized_stack(supervised, embdding_size)
        all_prob_supervised = transform_to_log_prob(x, zero_prob=zero_prob)
        x = optimized_stack(clm, embdding_size)
        all_prob_clm = transform_to_log_prob(x, zero_prob=zero_prob)

    temp_dict["input_ids"] = input_ids
    temp_dict["valid_label_index_list"] = valid_label_index_list
    temp_dict["all_prob_supervised"] = all_prob_supervised
    temp_dict["all_prob_clm"] = all_prob_clm
    temp_dict["supervised_cnt"] = supervised_cnt
    temp_dict["clm_cnt"] = clm_cnt
    return temp_dict


class SpecialDataset(Dataset):
    def __init__(
        self,
        synthesis_dict: dict[str : tuple[list[Counter], list[Counter]]],
        cnt_list: list[list[tuple]],  # 在单轮对话里每个list里应该只有tuple
        embedding_size,
        zero_prob=0.1,
        div_mode=False,
    ):

        synthesis_dict = [data_sample for data_sample in synthesis_dict.items()]
        self.supervised = [[synthesis_dict[i][1][j][0] for j in range(len(synthesis_dict[i][1]))] 
                   for i in range(len(synthesis_dict))]
        # self.supervised = [synthesis_dict[i][1][j][0]  for i in range(len(synthesis_dict))  for j in range(len(synthesis_dict[i]))]
        self.clm = [[synthesis_dict[i][1][j][1] for j in range(len(synthesis_dict[i][1]))] 
                   for i in range(len(synthesis_dict))]
        self.input_ids=[synthesis_dict[i][0] for i in range(len(synthesis_dict))]
        
        self.valid_label_index_list = cnt_list
        self.embedding_size = embedding_size
        self.zero_prob = zero_prob
        self.div_mode = div_mode

    def __getitem__(self, index):
        
        return {
            "input_ids":self.input_ids[index],
            "supervised":self.supervised[index],
            "clm":self.clm[index],
            "embedding_size":self.embedding_size,
            "zero_prob":self.zero_prob,
            "div_mode":self.div_mode,
            "valid_label_index_list":self.valid_label_index_list[index],
        }
        
        return get_data(
            self.supervised[index],
            self.clm[index],
            self.input_ids[index],
            self.cnt_list[index],
            self.embedding_size,
            zero_prob=self.zero_prob,
            div_mode=self.div_mode,
        )

    def __len__(self):
        return len(self.input_ids)


from torch.nn.utils.rnn import pad_sequence


class SpecialDataCollator:
    def __init__(self, tokenizer,zero_prob,embedding_size) -> None:
        self.tokenizer = tokenizer
        self.zero_prob = zero_prob
        self.embedding_size=embedding_size
        
        
    def __call__(self, batch) -> torch.Any:


        
        input_ids = [list(d["input_ids"]) for d in batch]
        input_ids_len = list(len(input_id) for input_id in input_ids)
        # input_ids_max_len = max(input_ids_len)
        input_ids = self.tokenizer.pad(
            {"input_ids": input_ids}, return_tensors="pt", padding=True
        ) 
        input_ids_max_len=input_ids['input_ids'].shape[-1]
        
        

        valid_label_index_list = (
            []
        )  # 这个东西很复杂……，因为pad之后前面会变长，所以前面还要去掉pad的位置。
        for i, d in enumerate(batch):
            length_diff = input_ids_max_len - input_ids_len[i]
            for j in range(len(d["valid_label_index_list"])):
                d["valid_label_index_list"][j] = (
                    length_diff + d["valid_label_index_list"][j][0],
                    length_diff + d["valid_label_index_list"][j][1],
                )
            valid_label_index_list.append(d["valid_label_index_list"])
        
        
        # debug
        
        # total_sum = 0
        # for sublist in valid_label_index_list:
        #     for pair in sublist:
        #         difference = abs(pair[1] - pair[0])
        #         total_sum += difference
 
        
        
        
        
        # all_prob_supervised = [d["all_prob_supervised"] for d in batch]
        # all_prob_clm = [d["all_prob_clm"] for d in batch]
        # supervised_cnt = [d["supervised_cnt"] for d in batch]
        # clm_cnt = [d["clm_cnt"] for d in batch]

        
        # 相较于input_ids，我们解开了对每个元素的list包围，使得一个batch的target从bsz，seqlen坍缩成了bsz x seqlen

        
        supervised= [item for d in batch for item in d['supervised'] ]
        clm=[item for d in batch for item in d['clm'] ]
        
        
        x = optimized_stack(supervised,self.embedding_size)

        all_prob_supervised = (1 - self.zero_prob) * x / torch.sum(x, dim=-1, keepdim=True)
        non_zero_cnt = torch.sum(x != 0, keepdim=True, dim=-1)
        temp_zero_prob = self.zero_prob / (self.embedding_size - non_zero_cnt)
        all_prob_supervised = torch.where(
            all_prob_supervised == 0, temp_zero_prob, all_prob_supervised
        )
        
        x = optimized_stack(clm, self.embedding_size)
        all_prob_clm = (1 - self.zero_prob) * x / torch.sum(x, dim=-1, keepdim=True)
        zero_cnt = torch.sum(x != 0, keepdim=True, dim=-1)
        temp_zero_prob = self.zero_prob / (self.embedding_size - zero_cnt)
        all_prob_clm = torch.where(all_prob_clm == 0, temp_zero_prob, all_prob_clm)
        
        supervised_cnt = torch.tensor([frequency(sum(xx.values()), xmax=10) for xx in supervised])
        clm_cnt =  torch.tensor([frequency(sum(xx.values())) for xx in clm])
       

        return {
            "input_ids": input_ids.input_ids,
            "attention_mask": input_ids.attention_mask,
            "all_prob_supervised": all_prob_supervised,
            "all_prob_clm": all_prob_clm,
            "valid_label_index_list": valid_label_index_list,
            "supervised_cnt": supervised_cnt,
            "clm_cnt": clm_cnt,
        }

if __name__ == '__main__':
    collator = SpecialDataCollator()
    train_dataset = load_dataset()
    dataloader = DataLoader(
        dataset=train_dataset, batch_size=8, collate_fn=collator, num_workers=60,pin_memory=False
    )

    from tqdm import tqdm

    for d in tqdm(dataloader):
        continue
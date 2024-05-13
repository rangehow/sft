from collections import Counter
from functools import partial
from torch.utils.data import Dataset
import torch

from scipy.optimize import fsolve


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

            def jacobian(x):
                # 自己写的呀科比矩阵比他自动推导的还慢，流汗黄豆了家人们
                tensor_with_temperature = knns / x
                exp_tensor = torch.exp(tensor_with_temperature)  # e^xi/T
                special_tensor = knns / (
                    x**2
                )  # xi/T^2 (本来这里应该有个负号的，但是被分子m之前的负号抵消了)
                factor = (
                    torch.mul(exp_tensor, special_tensor).sum(dim=1)
                    * zero_count_per_tensor
                )
                sum_exp = torch.sum(exp_tensor, dim=-1)
                square_sum = torch.square(sum_exp)
                return [torch.sum(sum_exp / square_sum)]

            def fun(x):
                if x <= 0:
                    return -10000
                
                tensor_with_temperature = knns / x

                exp_tensor = torch.exp(tensor_with_temperature)
                # nonzero_exp_tensor=exp_tensor[non_zero_index[:,0],non_zero_index[:,1]]
                # sum_exp = torch.sum(nonzero_exp_tensor,dim=-1)
                sum_exp = torch.sum(exp_tensor, dim=-1)
                result = torch.sum(zero_count_per_tensor / sum_exp) - bsz * zero_prob
                return result

            record = False
            # import pdb
            # pdb.set_trace()
            knn_temperature, info, status, message = fsolve(
                fun, 0.07, full_output=True, col_deriv=True
            )
            # x=symbols('x')
            # f=Function(fun)
            # knn_temperature = solve(Eq(fun(x), 0),x )
            # 2\32\28\3\0.85\4.59\0.51\1.49\49\1.68

            if status in [2, 3, 4, 5]:
                knn_temperature, info, status, message = fsolve(
                    fun,
                    500,
                    full_output=True,
                    col_deriv=True,
                )

            if status in [2, 3, 4, 5]:
                knn_temperature, info, status, message = fsolve(
                    fun, 0.03, full_output=True, col_deriv=True
                )

            if status in [2, 3, 4, 5]:
                initial_start_point = 1
                start_point = initial_start_point
                end_point = 50
                interval = max((end_point - start_point) // 5, 1)

                while status in [2, 3, 4, 5]:
                    knn_temperature, info, status, message = fsolve(
                        fun, start_point, full_output=True, col_deriv=True
                    )
                    # print(start_point,interval,initial_start_point,end_point,knn_temperature)

                    start_point += interval
                    if start_point > end_point:
                        print(start_point, info, knn_temperature)
                        if knn_temperature < 0:
                            knn_temperature = 1
                        # import pdb
                        # pdb.set_trace()
                        print("失败了，没找到温度")
                        break
        probs = torch.nn.functional.softmax(knns / knn_temperature, dim=-1)


    return probs


def frequency(x, xmax=50):
    return (x / xmax) ** 0.75 if x < xmax else 1


def get_data(
    synthesis_dict,
    valid_label_index_list,
    embdding_size,
    tokenizer,
    zero_prob,
    div_mode,
):
    # import time
    # a=time.time()
    temp_dict = {}
    supervised = [synthesis_dict[1][i][0] for i in range(len(synthesis_dict[1]))]
    clm = [synthesis_dict[1][i][1] for i in range(len(synthesis_dict[1]))]
    supervised_cnt = list(
        map(partial(frequency, xmax=10), [sum(xx.values()) for xx in supervised])
    )
    clm_cnt = list(map(frequency, [sum(xx.values()) for xx in clm]))

    if div_mode:
        

        x = torch.stack(
            [
                torch.bincount(
                    torch.tensor(list(xx.elements())), minlength=embdding_size
                )
                for xx in supervised
            ]
        )
        all_prob_supervised = x / torch.sum(x, dim=-1, keepdim=True)
        all_prob_supervised=(1-zero_prob)*x/torch.sum(x, dim=-1, keepdim=True)
        zero_cnt = torch.sum(x != 0,keepdim=True,dim=-1)
        temp_zero_prob=zero_prob / (embdding_size-zero_cnt)
        all_prob_supervised=torch.where(all_prob_supervised==0, temp_zero_prob , all_prob_supervised)

        x = torch.stack(
            [
                torch.bincount(
                    torch.tensor(list(xx.elements())), minlength=embdding_size
                )
                for xx in clm
            ]
        )
        # all_prob_supervised = x / torch.sum(x, dim=-1, keepdim=True)
        all_prob_clm=(1-zero_prob)*x/torch.sum(x, dim=-1, keepdim=True)
        zero_cnt = torch.sum(x != 0,keepdim=True,dim=-1)
        temp_zero_prob=zero_prob / (embdding_size-zero_cnt)
        all_prob_clm=torch.where(all_prob_clm==0, temp_zero_prob , all_prob_clm)


    else:
        # TODO 记得要统计一下 sum(clm[i].values())

        x = torch.stack(
            [
                torch.bincount(
                    torch.tensor(list(xx.elements())), minlength=embdding_size
                )
                for xx in supervised
            ]
        )

        all_prob_supervised = transform_to_log_prob(x, zero_prob=zero_prob)

        try:
            x = torch.stack(
                [
                    torch.bincount(
                        torch.tensor(list(xx.elements())), minlength=embdding_size
                    )
                    for xx in clm
                ]
            )

            all_prob_clm = transform_to_log_prob(x, zero_prob=zero_prob)
        except:
            print(supervised)
            print(clm)
            print([list(xx.elements()) for xx in clm])
            print([torch.tensor(list(xx.elements())) for xx in clm])

    

    temp_dict["input_ids"] = synthesis_dict[0]
    temp_dict["valid_label_index_list"] = valid_label_index_list
    temp_dict["all_prob_supervised"] = all_prob_supervised
    temp_dict["all_prob_clm"] = all_prob_clm
    temp_dict["supervised_cnt"] = supervised_cnt
    temp_dict["clm_cnt"] = clm_cnt
    # print('get_time',time.time()-a)
    return temp_dict


class SpecialDataset(Dataset):
    def __init__(
        self,
        synthesis_dict: dict[str : tuple[list[Counter], list[Counter]]],
        cnt_list: list[list[tuple]],  # 在单轮对话里每个list里应该只有tuple
        embedding_size,
        tokenizer=None,
        zero_prob=0.1,
        div_mode=False,
    ):

        self.synthesis_dict = [data_sample for data_sample in synthesis_dict.items()]
        self.cnt_list = cnt_list
        self.embedding_size = embedding_size
        self.zero_prob = zero_prob
        self.tokenizer = tokenizer
        self.div_mode = div_mode

    def __getitem__(self, index):

        return get_data(
            self.synthesis_dict[index],
            self.cnt_list[index],
            self.embedding_size,
            tokenizer=self.tokenizer,
            zero_prob=self.zero_prob,
            div_mode=self.div_mode,
        )
    def __len__(self):
        return len(self.synthesis_dict)


from torch.nn.utils.rnn import pad_sequence


class SpecialDataCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch) -> torch.Any:

        # import time
        # a=time.time()

        input_ids = [list(d["input_ids"]) for d in batch]
        input_ids_len=list(len(input_id) for input_id in input_ids)
        input_ids_max_len=max(input_ids_len)
        input_ids = self.tokenizer.pad(
            {"input_ids": input_ids}, return_tensors="pt", padding=True
        )
        
        all_prob_supervised = [d["all_prob_supervised"] for d in batch]
        all_prob_clm = [d["all_prob_clm"] for d in batch]
        valid_label_index_list = [] # 这个东西很复杂……，因为pad之后前面会变长，所以前面还要去掉pad的位置。
        for i,d in enumerate(batch):
            length_diff=input_ids_max_len-input_ids_len[i]
            for i in range(len(d['valid_label_index_list'])):
                d['valid_label_index_list'][i]=(length_diff+d['valid_label_index_list'][i][0],length_diff+d['valid_label_index_list'][i][1])
            valid_label_index_list.append(d['valid_label_index_list'])

        supervised_cnt = [d["supervised_cnt"] for d in batch]
        clm_cnt = [d["clm_cnt"] for d in batch]


        # print('collate time',time.time()-a)
        return {
            "input_ids": input_ids.input_ids,
            "attention_mask": input_ids.attention_mask,
            "all_prob_supervised": torch.cat(all_prob_supervised),
            "all_prob_clm": torch.cat(all_prob_clm),
            "valid_label_index_list": valid_label_index_list,
            "supervised_cnt": supervised_cnt,
            "clm_cnt": clm_cnt,
        }

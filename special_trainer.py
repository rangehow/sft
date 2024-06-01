from transformers import Seq2SeqTrainer, Trainer
import torch
import os
from torch.nn import KLDivLoss, CrossEntropyLoss
import torch.nn.functional as F
import time


class KLTrainer(Trainer):

    def __init__(self, weight_mode=False, **kwargs):
        self.weight_mode = weight_mode
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):

        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop("attention_mask")

        valid_label_index_list = inputs.pop(
            "valid_label_index_list"
        )  # 3dim list -> (batch_size, turn , 2) 只有turn才会产生一个起止下标

        all_prob_supervised = inputs.pop("all_prob_supervised")
        all_prob_clm = inputs.pop("all_prob_clm")
        supervised_cnt = inputs.pop("supervised_cnt")
        clm_cnt = inputs.pop("clm_cnt")

        # print(clm_cnt.dtype)
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        model_logits = result.logits  # bsz x seqlen x dim

        # NOTE 正确性检查见本文件底部test code 1
        last_logits = torch.cat(
            [
                row[start:end]
                for row, turn in zip(model_logits, valid_label_index_list)
                for start, end in turn
            ]
        ).to(model_logits.device)
        all_prob_supervised = all_prob_supervised.to(model_logits.device)
        all_prob_clm = all_prob_clm.to(model_logits.device)

        if not self.weight_mode:
            try:
                ce_loss = CrossEntropyLoss(ignore_index=-100)
                supervised_loss = ce_loss(last_logits, all_prob_supervised)
                clm_loss = ce_loss(last_logits, all_prob_clm)
            except:
                import pdb

                pdb.set_trace()
        else:
            ce_loss = CrossEntropyLoss(ignore_index=-100, reduction="none")
            # print(supervised_cnt.dtype, ce_loss(last_logits, all_prob_supervised).dtype)
            # supervised_loss = supervised_cnt.to(torch.float32) @ ce_loss(
            #     last_logits, all_prob_supervised
            # )
            # clm_loss = clm_cnt.to(torch.float32) @ ce_loss(last_logits, all_prob_clm)
            supervised_loss = supervised_cnt  @ ce_loss(
                last_logits, all_prob_supervised
            )
            clm_loss = clm_cnt  @ ce_loss(last_logits, all_prob_clm)

        if not self.weight_mode:
            loss = 0.8 * supervised_loss + 0.2 * clm_loss
        else:
            loss = (0.8 * supervised_loss + 0.2 * clm_loss) / last_logits.size()[0]
        # print('valid_label_index_list',valid_label_index_list)
        # print("supervised_loss", supervised_loss)
        # print("clm_loss", clm_loss)
        # print("loss", loss)
        if return_outputs:
            return loss, {"logits": model_logits}
        else:
            return loss

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys,
    ):
        model.eval()
        with torch.inference_mode():
            with self.compute_loss_context_manager():
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

        return (loss, None, None)


# test code 1:
# import torch

# # 假设你的张量名为tensor，索引范围名为indices
# model_logits = torch.stack((torch.arange(120),torch.arange(120))).unsqueeze(-1)

# print(model_logits.shape)
# valid_label_index_list=[[[57, 120]], [[38, 68]]]


# # 使用索引操作取出第二维的元素
# result=torch.cat([row[start:end+1] for row, turn in zip(model_logits, valid_label_index_list) for start, end in turn])
# print(result,result.shape)

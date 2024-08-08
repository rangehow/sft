from transformers import Seq2SeqTrainer, Trainer
import torch
import os
from torch.nn import KLDivLoss, CrossEntropyLoss
import torch.nn.functional as F
import time


class KDTrainer(Trainer):

    def __init__(self, teacher_model, temperature=1.0, alpha=0.5, **kwargs):
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        print("model's device",model.device)
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop("attention_mask")
        labels = inputs.pop("labels")

        student_outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        student_logits = student_outputs.logits  # batch_size x seq_len x vocab_size

        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=input_ids.to(self.teacher_model.device),
                attention_mask=attention_mask.to(self.teacher_model.device),
            )
            teacher_logits = teacher_outputs.logits.to(student_logits.device)

        # 创建标签掩码
        label_mask = (labels != -100).float()

        # 计算蒸馏损失
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction='none'
        )
        kd_loss = kd_loss.sum(-1) * (self.temperature ** 2)
        kd_loss = (kd_loss * label_mask).sum() / label_mask.sum()

        # 计算标准交叉熵损失
        ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), 
                                  labels.view(-1), 
                                  ignore_index=-100, 
                                  reduction='mean')

        # 组合损失
        loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss

        if return_outputs:
            return loss, {"logits": student_logits}
        else:
            return loss
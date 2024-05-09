from transformers import Seq2SeqTrainer,Trainer
import torch
import os
from torch.nn import KLDivLoss,CrossEntropyLoss
import torch.nn.functional as F


class KLTrainer(Trainer):
    def __init__(self,ce,**kwargs):
        self.ce=ce
        super().__init__(**kwargs)
        
    def compute_loss(self, model, inputs, return_outputs=False):
        
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop("attention_mask")
        decoder_input_ids = inputs.pop("decoder_input_ids")
        logit_index = inputs.pop("logit_index")
        # decoder_attention_mask = inputs.pop("decoder_attention_mask")
        target = inputs.pop("target")
        # debug_target = inputs.pop("debug_target")
        
        
        
        # print(attention_mask )
        result = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            # decoder_attention_mask=decoder_attention_mask,
        )
        model_logits=result.logits
        # ---------------------debug area
        # print(model_logits.shape)
        # print(target.shape)

        # loss_fct = CrossEntropyLoss(ignore_index=-100)
        # # move labels to correct device to enable PP
        # aaaa = torch.cat((decoder_input_ids[:,1:],torch.ones((decoder_input_ids.size(0),1),device=decoder_input_ids.device,dtype=torch.int64)),dim=-1)
        # print(debug_target)
        # lossss = loss_fct(model_logits.view(-1, model_logits.size(-1)), debug_target.view(-1))
        
        
        
        
        # print('target的deivce在那里？',target.device)
        # 性能优化前：
        # last_logits = torch.zeros((model_logits.shape[0], model_logits.shape[-1]))
        # for i, index in enumerate(logit_index):
        #     last_logits[i] = model_logits[i, index]
        # last_logits = last_logits.squeeze()
        # 性能优化后：
        # last_logits =model_logits[torch.arange(model_logits.size(0)), logit_index].squeeze()
        # 最后会形成一个 需要计算loss的词 * vocab_size的分布，batch里不同的话按顺序平铺
        last_logits =torch.cat([row[:index+1] for row, index in zip(model_logits, logit_index)])
        
        # print(last_logits.shape)
        # import pdb
        # pdb.set_trace()
        # print('-------------------------')
        # print(torch.topk(F.softmax(last_logits, dim=-1),5,))
        # # print(decoder_input_ids)
        # print(torch.topk(target,k=5))
        # print('-------------------------')
        target = target.to(last_logits.device)
        # print('target的deivce在那里？',target.device)
        # last_logits = last_logits.to(model_logits.device)
        # print (last_logits==model_logits[:,-1])
        # print('last_logits.shape',last_logits.shape)
        # print('target.shape',target.shape)
        # print('---')
        
        # kl part---------------------------------------------------------
        # kl_loss = KLDivLoss(reduction="batchmean")
        # loss = kl_loss(F.log_softmax(last_logits, dim=-1), target).to(
        #     model_logits.device
        # )
        # ---------------------------------------------------------------
        # import pdb
        # pdb.set_trace()
        if self.ce:
            ce_loss=CrossEntropyLoss(ignore_index=-100)
            loss = ce_loss(last_logits,target).to(
                model_logits.device
            )
        else:
            kl_loss = KLDivLoss(reduction="batchmean")
            loss = kl_loss(F.log_softmax(last_logits, dim=-1), target).to(
                model_logits.device
            )
        
        # print('lossss',lossss,'loss',loss)
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

    # def save_model(self, output_dir, _internal_call):
    #     # if not os.path.exists(output_dir):
    #     #     os.makedirs(output_dir)

    #     # torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
    #     self.model.save_pretrained(output_dir)
    #     # self.tokenizer.save_pretrained(output_dir)

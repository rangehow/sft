"""
https://github.com/google-deepmind/gemma/blob/main/colabs/gsm8k_eval.ipynb
"""

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
from ..dataset_func import dname2func
from ..template import modelType2Template
from transformers import AutoTokenizer, AutoConfig, DataCollatorForSeq2Seq
import os
from .post_process import dname2post
from torch.utils.data import DataLoader
from .load_func import dname2load
from .samplingparam import dname2samplingparams

def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        type=int,
        default=0,
        help="0: base model(wo chat template),1:instruct model",
    )
    parser.add_argument(
        "--vllm",
        action="store_true",
    )
    parser.add_argument(
        "--shot",
        action="store_true",
    )
    parser.add_argument(
        "--dataset",
    )
    parser.add_argument("--model")
    parser.add_argument(
        "--dp",
        action="store_true",
    )
    parser.add_argument(
        "--logprob",
        action="store_true",
    )
    
    return parser.parse_args()


@logger.catch
def main():
    args = parse_args()

    model_type = AutoConfig.from_pretrained(
        os.path.join(args.model, "config.json")
    ).model_type
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    template = modelType2Template[model_type](tokenizer)

    dataset = dname2load[args.dataset]()

    test_dataset = dataset.map(
        partial(
            dname2func[args.dataset],
            template=template,
            test=True,
            shot=args.shot,
            vllm=args.vllm,
            mode=args.mode,
        ),
        batched=True,
        num_proc=1,  # 进程数不要设置太大，我不知道datasets咋设计的，进程数太大很慢
        desc="tokenize",
        load_from_cache_file=False,
    )

    print(test_dataset[0])
    # for i in test_dataset:
    #     print('question',i['question'])
    #     print('answer',i['answer'])
    #     print('input_ids\n',i['input_ids'])
    #     print('-'*20)
    #     import pdb
    #     pdb.set_trace()
    if args.vllm:
        from vllm import LLM 

        script_path = os.path.dirname(os.path.abspath(__file__))
        print("script_path", script_path)
        
        model_name = os.path.basename(args.model.rstrip(os.sep)) # 不去掉sep，碰到 a/b/ 就会读到空。
        save_str = (
            f"{model_name}_{args.dataset}_vllm_{args.mode}_shot"
            if args.shot
            else f"{model_name}_{args.dataset}_vllm_{args.mode}"
        )
        print("save_str", save_str)
        target_file = os.path.join(script_path, "generated", save_str)
        print("target_file", target_file)
        reuse_flag = False
        if os.path.exists(target_file):
            while True:
                i = input("本次任务似乎已经被完成过了~输入y可以复用，输入n则重新生成：")
                if i == "y":
                    reuse_flag = True
                    break
                elif i == "n":
                    break
                else:
                    print("输入错误，必须是y或n")

        if reuse_flag:
            with open(target_file, "rb") as r:
                response = pickle.load(r)
        else:
            
            all_prompt = [d["input_ids"] for d in test_dataset]
            samplingParams=dname2samplingparams[args.dataset]()
            if args.dp:
                def split_list(lst, n=torch.cuda.device_count()):
                    avg = len(lst) / float(n)
                    return [lst[int(avg * i):int(avg * (i + 1))] for i in range(n)]
                all_prompt=split_list(all_prompt)

                import ray
                @ray.remote(num_gpus=1)
                def run(prompts):
                    model = LLM(model=args.model)
                    response = model.generate(prompts, samplingParams)
                    return response

                outputs=[]
                for i in range(len(all_prompt)):
                    output=run.remote(all_prompt[i])
                    outputs.append(output)

                response=[]
                for i in range(len(outputs)):
                    result=ray.get(outputs[i])
                    response.extend(result)

                
                
                ray.shutdown()


            else:
                model = LLM(model=args.model,tensor_parallel_size=torch.cuda.device_count() )
                
                if args.logprob:
                    response = []
                    # all_prompt内容大概长这样，每一个列表的列表对应一个问题和它对应的选项。[[[问题1+选项1],[问题1+选项2]],[[问题2+选项1],[问题2+选项2]]
                    for input in all_prompt:
                        res = []
                        for ins in input:
                                
                            # 对于每一个问题+选项生成一个输出 
                            output = model.generate(
                                    prompt_token_ids=[ins],
                                    sampling_params=samplingParams
                                )

                            res.append(output)
                             
                        response.append(res)
                        
                else:
                    response = model.generate(all_prompt, samplingParams)
                
                
            logger.debug(f"response的长度:{len(response)}")
            # 不只保存文本是因为未来很可能有一些任务，是需要log prob的，所以没办法，最好整个保存。
            with open(target_file, "wb") as o:
                pickle.dump(response, o)

    else:
        from transformers import (
            AutoModelForCausalLM,
            GenerationConfig,
            StoppingCriteria,
        )

        class DataCollator:
            def __init__(self, tokenizer):
                self.tokenizer = tokenizer

            def __call__(
                self,
                instance,
            ):

                inputs = self.tokenizer(
                    [i["input_ids"] for i in instance],
                    padding=True,
                    return_tensors="pt",
                    pad_to_multiple_of=8,
                )
                return {
                    "input_ids": inputs.input_ids,
                    "attention_mask": inputs.attention_mask,
                }

        with torch.inference_mode():
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    attn_implementation="flash_attention_2",
                )
            except Exception as e:
                logger.error(e)
                logger.error("尝试退回naive attn，如果torch>2.1则是sqpa")
                model = AutoModelForCausalLM.from_pretrained(
                    args.model,
                    torch_dtype=torch.bfloat16,
                )

            model.cuda()

            collator = DataCollator(tokenizer=tokenizer)
            dataloader = DataLoader(
                dataset=test_dataset, collate_fn=collator, batch_size=1, num_workers=8
            )
            response = []

            class EosListStoppingCriteria(StoppingCriteria):
                def __init__(self, eos_sequence):
                    self.eos_sequence = eos_sequence

                def __call__(
                    self,
                    input_ids: torch.LongTensor,
                    scores: torch.FloatTensor,
                    **kwargs,
                ) -> bool:

                    last_ids = input_ids[:, -len(self.eos_sequence) :].tolist()
                    return self.eos_sequence in last_ids

            from tqdm import tqdm

            for d in tqdm(dataloader):
                prompt_length = d["input_ids"].shape[1]
                generation_config = GenerationConfig(
                    max_new_tokens=1024,
                    do_sample=False,
                )

                output = model.generate(
                    input_ids=d["input_ids"].to("cuda"),
                    attention_mask=d["attention_mask"].to("cuda"),
                    generation_config=generation_config,
                    tokenizer=tokenizer,
                    stopping_criteria=[
                        EosListStoppingCriteria(
                            tokenizer.encode("Q:", add_special_tokens=False)
                        )
                    ],
                )
                text = tokenizer.batch_decode(output[:, prompt_length:])
                print(text)
                response.append(text)

    score = dname2post[args.dataset](
        prediciton=response,
        reference=[t["answer"] for t in test_dataset],
        vllm=args.vllm,
    )

    logger.debug(f"task:{args.dataset},model:{args.model},score :{score}")


if __name__ == "__main__":
    main()

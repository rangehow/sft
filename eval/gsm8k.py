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

    if args.vllm:
        from vllm import LLM, SamplingParams

        script_path = os.path.dirname(os.path.abspath(__file__))
        print("script_path", script_path)
        model_name = os.path.basename(args.model)
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
                i = input("本次任务似乎已经被完成过了~输入y可以复用，输入n则重新生成")
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
            # print('将在这么多卡上张量并行:',torch.cuda.device_count())
            model = LLM(model=args.model, swap_space=0, gpu_memory_utilization=0.9)
            samplingParams = SamplingParams(
                max_tokens=1024,
                temperature=0,
                stop=["Q:"],
            )

            all_prompt = [d["input_ids"] for d in test_dataset]

            response = model.generate(all_prompt, samplingParams)
            print("response的长度", len(response))
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

    print(f"score :{score}")


if __name__ == "__main__":
    main()

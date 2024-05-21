"""
https://github.com/google-deepmind/gemma/blob/main/colabs/gsm8k_eval.ipynb
"""

from functools import partial
import json
import datasets
import re
from loguru import logger
import os
from argparse import ArgumentParser
import torch
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
    parser.add_argument("--output")
    return parser.parse_args()


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


# if os.path.exists(args.output):
#     logger.error(f"{args.output}已经存在")
#     exit()


# with open(args.output, "w", encoding="utf-8") as o:

if args.vllm:
    from vllm import LLM, SamplingParams

    model = LLM(model=args.model, swap_space=0)
    samplingParams = SamplingParams(
        max_tokens=1024, temperature=0, logprobs=5, stop=["Q:"]
    )
    all_prompt = [d["input_ids"] for d in test_dataset]
    response = model.generate(all_prompt, samplingParams)

else:
    from transformers import AutoModelForCausalLM, GenerationConfig, StoppingCriteria

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
                self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
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
    prediciton=response, reference=[t["answer"] for t in test_dataset], vllm=args.vllm
)

print(f"score :{score}")
# json.dump(all_responses, o, ensure_ascii=False, indent=4)

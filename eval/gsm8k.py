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
from transformers import AutoTokenizer,AutoConfig
import os




def find_numbers(x: str) -> list[str]:
    """Finds all numbers in a string."""
    # Search for number, possibly negative (hyphen), with thousand separators
    # (comma), and with a decimal point (period inbetween digits).
    numbers = re.compile(
        r"-?[\d,]*\.?\d+",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    ).findall(x)
    return numbers


def find_number(x: str, answer_delimiter: str = "The answer is") -> str:
    """Finds the most relevant number in a string."""
    # If model uses the answer delimiter, then select the first number following
    # that format.
    if answer_delimiter in x:
        answer = x.split(answer_delimiter)[-1]
        numbers = find_numbers(answer)
        if numbers:
            return numbers[0]

    # In general, select the last number in the string.
    numbers = find_numbers(x)
    if numbers:
        return numbers[-1]
    return ""


def maybe_remove_comma(x: str) -> str:
    # Example: 5,600 -> 5600
    return x.replace(",", "")





all_correct = 0
all_responses = {}
short_responses = {}
idx = 0
correct = 0

TEMPLATE = """
Q: {question}
A:"""



def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode", type=int,default=0, help="0: base model(wo chat template),1:instruct model"
    )
    parser.add_argument(
        "--vllm",
        action="store_true",
    )
    parser.add_argument(
        "--shot",
        action="store_true",
    )
    parser.add_argument('--dataset',)
    parser.add_argument("--model")
    parser.add_argument("--output")
    return parser.parse_args()


args = parse_args()



model_type=AutoConfig.from_pretrained(os.path.join(args.model,'config.json')).model_type
tokenizer=AutoTokenizer.from_pretrained(args.model)
tokenizer.padding_side='left'
template=modelType2Template[model_type](tokenizer)

dataset = datasets.load_dataset(args.dataset,'main')['test']
test_dataset = dataset.map(
        partial(dname2func[args.dataset], template=template,test=True,shot=args.shot,vllm=args.vllm,mode=args.mode),
        batched=True,
        num_proc=1, # 进程数不要设置太大，我不知道datasets咋设计的，进程数太大很慢
        desc="tokenize",
        load_from_cache_file=False,
    )

import pdb
pdb.set_trace()
if os.path.exists(args.output):
    logger.error(f"{args.output}已经存在")
    exit()

with open(args.output, "w", encoding="utf-8") as o:

    if args.vllm:
        from vllm import LLM, SamplingParams
        model = LLM(model=args.model)
        samplingParams = SamplingParams(max_tokens=1024, top_k=1)
        all_prompt=[d['input_ids'] for d in test_dataset]
        response = model.generate(all_prompt, samplingParams)
        
    else:
        from transformers import  AutoModelForCausalLM
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2"
            )
        except Exception as e:
            logger.error(e)
            logger.error('尝试退回naive attn，如果torch>2.1则是sqpa')
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.bfloat16,
            )
        
    

    for task_id, output in enumerate(response):
        generated_text = output.outputs[0].text

        all_responses[task_id] = generated_text.split("\nQ:")[0]
        short_responses[task_id] = maybe_remove_comma(
            find_number(all_responses[task_id])
        )

        print(f"Short answer: {short_responses[task_id]}")

        try:
            correct += float(
                maybe_remove_comma(find_number(test_dataset[task_id]["answer"]))
            ) == float(short_responses[task_id])
        except:
            correct += maybe_remove_comma(
                find_number(test_dataset[task_id]["answer"])
            ) == maybe_remove_comma(find_number(short_responses[task_id]))
        print("-" * 40)
        print(f"Ground truth answer {test_dataset[task_id]['answer']}")
        print(
            f"Short ground truth answer {find_number(test_dataset[task_id]['answer'])}"
        )
        print(f"Correct: {correct} out of {idx+1}")
        print("=" * 40)
        idx += 1

    all_responses["score"] = correct / idx *100
    json.dump(all_responses, o, ensure_ascii=False, indent=4)

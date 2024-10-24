from functools import partial
import os
from typing import Literal
import torch
from vllm.sampling_params import SamplingParams
from outlines import models
import outlines
from pydantic import BaseModel
from loguru import logger
from argparse import ArgumentParser
from transformers import AutoTokenizer, AutoConfig
from .load_func import dname2load
from ..dataset_func import dname2func
from ..template import modelType2Template
from ..config import *


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        type=int,
        default=0,
        help="0: base model(wo chat template),1:instruct model",
    )
    parser.add_argument(
        "--shot",
        action="store_true",
    )
    parser.add_argument(
        "--reuse",
        action="store_true",
        help="在批量测试时自动开reuse",
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
    parser.add_argument(
        "--output_path",
    )
    parser.add_argument(
        "--timestamp",  # 不要用默认值，因为和lm eval的就不能放在一起了？
    )
    return parser.parse_args()


class my(BaseModel):
    
    answer: Literal["A", "B", "C", "D", "E"]


sampling_params = SamplingParams(
    temperature=0.0, max_tokens=1024, repetition_penalty=1.0
)


@logger.catch
def main():
    args = parse_args()

    model_list = args.model.split(",")
    dataset_list = args.dataset.split(",")

    # os.makedirs(args.output_path,exist_ok=True)
    script_path = os.path.dirname(os.path.abspath(__file__))  # gsm8k文件的所在目录

    print("script_path", script_path)

    for m in model_list:
        os.makedirs(os.path.join(os.path.dirname(args.output_path), m), exist_ok=True)
        model_type = AutoConfig.from_pretrained(m).model_type

        tokenizer = AutoTokenizer.from_pretrained(m)
        tokenizer.padding_side = "left"
        template = modelType2Template[model_type](tokenizer)
        model_name = os.path.basename(
            m.rstrip(os.sep)
        )  # 不去掉sep，碰到 a/b/ 就会读到空。
        record_list = []
        for d in dataset_list:

            dataset = dname2load[d](dataset_dir.get(d, None))
            test_dataset = dataset.map(
                partial(
                    dname2func[d],
                    template=template,
                    test=True,
                    mode=args.mode,
                ),
                batched=True,
                num_proc=1,  # 进程数不要设置太大，我不知道datasets咋设计的，进程数太大很慢
                desc="tokenize",
                load_from_cache_file=False,
                remove_columns=dataset.features.keys(),
            )

            print(test_dataset[0])

            save_str = (
                f"{model_name}_{d}_vllm_{args.mode}_shot"
                if args.shot
                else f"{model_name}_{d}_vllm_{args.mode}"
            )
            print("save_str", save_str)
            target_file = os.path.join(script_path, "generated", save_str)
            print("target_file", target_file)
            reuse_flag = True if args.reuse and os.path.exists(target_file) else False
            # if not args.reuse:
            #     if os.path.exists(target_file):
            #         while True:
            #             i = input(
            #                 "本次任务似乎已经被完成过了~输入y可以复用，输入n则重新生成："
            #             )
            #             if i == "y":
            #                 reuse_flag = True
            #                 break
            #             elif i == "n":
            #                 break
            #             else:
            #                 print("输入错误，必须是y或n")

            # if reuse_flag:
            #     with open(target_file, "rb") as r:
            #         response = pickle.load(r)
            # else:
            all_prompt = [d["input_ids"] for d in test_dataset]
            if args.dp:

                def split_list(lst, n=torch.cuda.device_count()):
                    avg = len(lst) / float(n)
                    return [lst[int(avg * i) : int(avg * (i + 1))] for i in range(n)]

                all_prompt = split_list(all_prompt)

                import ray

                @ray.remote(num_gpus=1)
                def run(prompts):
                    def available_memory_ratio():
                        import pynvml

                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        total = info.total
                        free = info.free
                        return free / total

                    print("可用的内存：", available_memory_ratio())

                    # print(schema,type(schema))

                    model = models.vllm(
                        m,
                        gpu_memory_utilization=0.9 * available_memory_ratio(),
                    )
                    # generator = outlines.generate.json(model, my)
                    generator = outlines.generate.choice(
                        model, ["(A)", "(B)", "(C)", "(D)", "(E)"]
                    )
                    response = generator(
                        prompts,
                        sampling_params,
                    )

                    return response

                outputs = []
                for i in range(len(all_prompt)):
                    output = run.remote(all_prompt[i])
                    outputs.append(output)

                response = []
                for i in range(len(outputs)):
                    result = ray.get(outputs[i])
                    response.extend(result)

                ray.shutdown()

            else:

                model = models.vllm(
                    model=m,
                    tensor_parallel_size=torch.cuda.device_count(),
                    gpu_memory_utilization=0.9,
                    # enable_prefix_caching=True,
                )

                # if args.logprob:
                #     response = []
                #     # all_prompt内容大概长这样，每一个列表的列表对应一个问题和它对应的选项。[[[问题1+选项1],[问题1+选项2]],[[问题2+选项1],[问题2+选项2]]
                #     for input in all_prompt:
                #         res = []
                #         for ins in input:

                #             # 对于每一个问题+选项生成一个输出
                #             output = model.generate(
                #                 prompt_token_ids=[ins], sampling_params=samplingParams
                #             )

                #             res.append(output)

                #         response.append(res)

                # else:
                # max_len = 0
                # for p in all_prompt:
                #     max_len = max(max_len, len(p))
                # print(max_len)
                response = model.generate(all_prompt, sampling_params)

            logger.debug(f"response的长度:{len(response)}")
            # 不只保存文本是因为未来很可能有一些任务，是需要log prob的，所以没办法，最好整个保存。
            # os.makedirs(os.path.join(script_path, "generated"), exist_ok=True)
            # with open(target_file, "wb") as o:
            #     pickle.dump(response, o)

            score = 0
            for a, r in zip(response, [t["answer"] for t in test_dataset]):
                try:
                    if a == r:
                        score += 1
                except Exception as e:
                    print(e)
                    print(a,r)
                    pass
            print(score / len(response) * 100)
            import pdb

            pdb.set_trace()
            logger.debug(f"task:{d},model:{m},score :{score}")

            record_list.append(f"task:{d},model:{m},score :{score}")
        try:
            with open(
                os.path.join(
                    os.path.dirname(args.output_path), m, f"{args.timestamp}.jsonl"
                ),
                "w",
                encoding="utf-8",
            ) as o:
                json.dump(record_list, o, indent=2, ensure_ascii=False)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            os.makedirs(
                os.path.join(os.path.dirname(args.output_path), m), exist_ok=True
            )
            with open(
                os.path.join(
                    os.path.dirname(args.output_path), m, f"{args.timestamp}.jsonl"
                ),
                "w",
                encoding="utf-8",
            ) as o:
                json.dump(record_list, o, indent=2, ensure_ascii=False)
        output_str = os.path.join(
            os.path.dirname(args.output_path), m, f"{args.timestamp}.jsonl"
        )
        logger.debug(f"model:{m}的结果已保存至{output_str}")


if __name__ == "__main__":
    main()

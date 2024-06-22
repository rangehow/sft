import re
import datasets

dname2load = {}


def register2dict(name):
    def decorator(func):
        dname2load[name] = func
        return func

    return decorator


@register2dict(name="gsm8k")
def gsm8k(local_dir):
    if local_dir is not None:
        return datasets.load_dataset(local_dir, "main")["test"]
    
    return datasets.load_dataset("gsm8k", "main")["test"]


@register2dict(name="bbh")
def bbh(local_dir):
    if local_dir is not None:
        return datasets.load_dataset(local_dir)["validation"]
    
    return datasets.load_dataset("huanmit/flan-t5-boosting-bbh_cot")["validation"]


@register2dict(name="apps")
def apps(local_dir):
    if local_dir is not None:
        return datasets.load_dataset(local_dir, "all")["train"]
    return datasets.load_dataset("codeparrot/apps", "all")["train"]

@register2dict(name="alpaca_gpt4")
def alpaca_gpt4(local_dir):
    if local_dir is not None:
        return datasets.load_dataset(local_dir)["train"]
    return datasets.load_dataset("vicgalle/alpaca-gpt4")["train"]



@register2dict(name="mmlu")
def mmlu(local_dir):
    if local_dir is not None:
        return datasets.load_dataset(local_dir, "all")["test"]
    return datasets.load_dataset("cais/mmlu", "all")["test"]


@register2dict(name="humaneval")
def humaneval(local_dir):
    if local_dir is not None:
        return datasets.load_dataset(local_dir)['test']
    # from human_eval.data import write_jsonl, read_problems
    # problems = read_problems()

    # num_samples_per_task = 200
    # import pdb
    # pdb.set_trace()
    # samples = [
    #     dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    #     for task_id in problems
    #     for _ in range(num_samples_per_task)
    # ]
    return datasets.load_dataset('openai/openai_humaneval')['test']
    
@register2dict(name="truthfulqa")
def truthfulqa(local_dir):
    if local_dir is not None:
        return datasets.load_dataset(local_dir, "multiple_choice")['validation']
    else:
        return datasets.load_dataset('truthful_qa', "multiple_choice")['validation']

@register2dict(name="math")
def math(local_dir):
    if local_dir is not None:
        return datasets.load_dataset(local_dir,'all')['train']
    return datasets.load_dataset('lighteval/MATH','all')['train']

if __name__ == "__main__":
    humaneval()


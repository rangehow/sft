import re
import datasets

dname2load = {}


def register2dict(name):
    def decorator(func):
        dname2load[name] = func
        return func

    return decorator


@register2dict(name="gsm8k")
def gsm8k():
    return datasets.load_dataset("gsm8k", "main")["test"]


@register2dict(name="bbh")
def bbh():
    # available_subset = datasets.get_dataset_config_names("lukaemon/bbh")
    
    # return datasets.load_dataset("lukaemon/bbh")["test"]
    
    return datasets.load_dataset("huanmit/flan-t5-boosting-bbh_direct")["validation"]


@register2dict(name="apps")
def apps():
    return datasets.load_dataset("codeparrot/apps", "all")["train"]

@register2dict(name="alpaca_gpt4")
def alpaca_gpt4():
    return datasets.load_dataset("vicgalle/alpaca-gpt4")["train"]



@register2dict(name="mmlu")
def mmlu():
    return datasets.load_dataset("cais/mmlu", "all")["test"]


@register2dict(name="humaneval")
def humaneval():
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
def truthfulqa():
    return datasets.load_dataset('truthful_qa', "multiple_choice")['validation']

if __name__ == "__main__":
    humaneval()


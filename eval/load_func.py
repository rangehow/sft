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


@register2dict(name="mmlu")
def mmlu():
    return datasets.load_dataset("cais/mmlu", "all")["test"]
    
@register2dict(name="truthfulqa")
def truthfulqa():
    return datasets.load_dataset('truthful_qa', "multiple_choice")['validation']

if __name__ == "__main__":
    d=datasets.load_dataset("cais/mmlu")
    print(d)

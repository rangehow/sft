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
    return datasets.load_dataset("hails/mmlu_no_train", "all")["test"]


if __name__ == "__main__":
    # dd = mmlu()
    # print(dd)




    print(extract_answer("The answer is (B)1213A.\n\n"))
    print(find_matches("(B)12313A",'B'))
    # ss=set()
    # for d in dd['subject']:
    #     print(d)
    #     ss.add(d)
    # print(len(ss))
    # import json
    # a=open('/data/ruanjh/best_training_method/sft/eval/mmlu_cot_prompts.json')
    # aa=json.load(a)
    # print(len(aa))

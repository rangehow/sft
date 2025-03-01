import re
import datasets

dname2load = {}


def register2dict():
    def decorator(func):
        if func.__name__ not in dname2load:
            dname2load[func.__name__] = func
        else:
            print("重名了", func.__name__)
            exit()
        return func

    return decorator


@register2dict()
def gsm8k(local_dir, **kwargs):
    if local_dir is not None:
        return datasets.load_dataset(local_dir, "main")["test"]

    return datasets.load_dataset("gsm8k", "main")["test"]


@register2dict()
def bbh(local_dir, **kwargs):
    if local_dir is not None:
        return datasets.load_dataset(local_dir)["train"]

    return datasets.load_dataset("JesusCrist/bbh_cot_fewshot")["train"]


@register2dict()
def code(local_dir, **kwargs):
    if local_dir is not None:
        return datasets.load_dataset(local_dir)["train"]
    return datasets.load_dataset("m-a-p/CodeFeedback-Filtered-Instruction")["train"]


@register2dict()
def apps(local_dir, **kwargs):
    exit()  # '***数据集，不许用'
    if local_dir is not None:
        return datasets.load_dataset(local_dir, "all")["train"]
    return datasets.load_dataset("codeparrot/apps", "all")["train"]


@register2dict()
def alpaca_gpt4(local_dir, **kwargs):
    if local_dir is not None:
        return datasets.load_dataset(local_dir)["train"]
    return datasets.load_dataset("vicgalle/alpaca-gpt4")["train"]


@register2dict()
def mmlu(local_dir, **kwargs):
    if local_dir is not None:
        return datasets.load_dataset(local_dir, "all")["test"]
    return datasets.load_dataset("cais/mmlu", "all")["test"]


@register2dict()
def humaneval(local_dir, **kwargs):
    if local_dir is not None:
        return datasets.load_dataset(local_dir)["test"]
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
    return datasets.load_dataset("openai/openai_humaneval")["test"]


@register2dict()
def truthfulqa(local_dir, **kwargs):
    if local_dir is not None:
        return datasets.load_dataset(local_dir, "multiple_choice")["validation"]
    else:
        return datasets.load_dataset("truthful_qa", "multiple_choice")["validation"]


@register2dict()
def math(local_dir, **kwargs):
    if local_dir is not None:
        a = datasets.load_dataset(local_dir, "default")["train"]
        b = datasets.load_dataset(local_dir, "default")["test"]
    else:
        a = datasets.load_dataset("lighteval/MATH", "all")["train"]  # 7.5k
        b = datasets.load_dataset("lighteval/MATH", "all")["test"]  # 5k
    return datasets.concatenate_datasets([a, b])


@register2dict()
def slimpajama(local_dir, **kwargs):
    if local_dir is not None:
        return datasets.load_dataset(local_dir)["train"]
    else:
        return datasets.load_dataset("DKYoon/SlimPajama-6B")["train"]


@register2dict()
def test(local_dir, **kwargs):
    if local_dir is not None:
        return datasets.load_dataset(local_dir)["train"]
    else:
        return datasets.load_dataset("JesusCrist/SimpleData")["train"]


@register2dict()
def iepile(local_dir, **kwargs):
    if local_dir is not None:
        return datasets.load_dataset(local_dir)["train"]
    else:
        return datasets.load_dataset("JesusCrist/IEPILE")["train"]


@register2dict()
def wiki_medical(local_dir, **kwargs):
    if local_dir is not None:
        return datasets.load_dataset(local_dir)["train"]
    else:
        return datasets.load_dataset("gamino/wiki_medical_terms")["train"]


@register2dict()
def pubmed(local_dir, **kwargs):
    if local_dir is not None:
        return datasets.load_dataset(local_dir)["train"]
    else:
        return datasets.load_dataset(
            "JesusCrist/pubmed",
        )["train"]


@register2dict()
def pubmed_abstract(local_dir, **kwargs):
    if local_dir is not None:
        return datasets.load_dataset(local_dir)["train"]
    else:
        return datasets.load_dataset(
            "brainchalov/pubmed_arxiv_abstracts_data",
        )["train"]


@register2dict()
def medical_transcription(local_dir, **kwargs):
    if local_dir is not None:
        return datasets.load_dataset(local_dir)["train"]
    else:
        return datasets.load_dataset("rungalileo/medical_transcription_40")["train"]


@register2dict()
def textbooks(local_dir, **kwargs):
    if local_dir is not None:
        return datasets.load_dataset(local_dir)["train"]
    else:
        return datasets.load_dataset("MedRAG/textbooks")["train"]


@register2dict()
def medpt(local_dir, **kwargs):
    if local_dir is not None:
        return datasets.load_dataset(local_dir)["train"]
    else:
        return datasets.load_dataset("FreedomIntelligence/huatuo_encyclopedia_qa")[
            "train"
        ]


@register2dict()
def medqa(local_dir, **kwargs):
    if local_dir is not None:
        return datasets.load_dataset(local_dir)["test"]
    else:
        return datasets.load_dataset("truehealth/medqa")["test"]


@register2dict()
def medquad(local_dir, **kwargs):
    if local_dir is not None:
        return datasets.load_dataset(local_dir)["train"]
    else:
        return datasets.load_dataset("keivalya/MedQuad-MedicalQnADataset")["train"]


@register2dict()
def medmcqa(local_dir, **kwargs):
    if local_dir is not None:
        return datasets.load_dataset(local_dir)["validation"]
    else:
        return datasets.load_dataset("openlifescienceai/medmcqa")["validation"]


@register2dict()
def medical(local_dir, **kwargs):
    if local_dir is not None:
        return datasets.load_dataset(local_dir)["train"]
    else:
        return datasets.load_dataset("JesusCrist/med_qa_fewshot")["train"]


@register2dict()
def pubmedqa(local_dir, **kwargs):
    if local_dir is not None:
        return datasets.load_dataset(local_dir)["train"]
    else:
        return datasets.load_dataset("qiaojin/PubMedQA", "pqa_labeled")["train"]


@register2dict()
def bioasq(local_dir, **kwargs):
    if local_dir is not None:
        return datasets.load_dataset(local_dir)["test"]
    else:
        return datasets.load_dataset("Coldog2333/bioasq")["test"]


@register2dict()
def careqa(local_dir, **kwargs):
    if local_dir is not None:
        return datasets.load_dataset(local_dir)["test"]
    else:
        return datasets.load_dataset("HPAI-BSC/CareQA", "CareQA_en")["test"]


@register2dict()
def multimedqa(local_dir, **kwargs):
    if local_dir is not None:
        return datasets.load_dataset(local_dir)["test"]
    else:
        return datasets.load_dataset("openlifescienceai/multimedqa")["test"]


@register2dict()
def magpie(local_dir, **kwargs):
    if local_dir is not None:
        return datasets.load_dataset(local_dir)["train"]
    else:
        return datasets.load_dataset("Magpie-Align/MagpieLM-SFT-Data-v0.1")["train"]


@register2dict()
def magpie_300k(local_dir, **kwargs):
    if local_dir is not None:
        return datasets.load_dataset(local_dir)["train"]
    else:
        return datasets.load_dataset("Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered")[
            "train"
        ]


@register2dict()
def redpajama(local_dir, **kwargs):
    if local_dir is not None:
        return datasets.load_dataset(local_dir)["train"]
    else:
        return datasets.load_dataset("togethercomputer/RedPajama-Data-1T-Sample")[
            "train"
        ]


@register2dict()
def alma_zhen(local_dir, **kwargs):
    test_flag = kwargs.pop("test", False)
    if test_flag:
        if local_dir is not None:
            return datasets.load_dataset(local_dir, "zh-en")["validation"]
        else:
            return datasets.load_dataset("haoranxu/ALMA-Human-Parallel", "zh-en")[
                "validation"
            ]
    else:
        if local_dir is not None:
            return datasets.load_dataset(local_dir, "zh-en")["train"]
        else:
            return datasets.load_dataset("haoranxu/ALMA-Human-Parallel", "zh-en")[
                "train"
            ]


if __name__ == "__main__":
    d = datasets.load_dataset("/mnt/rangehow/rangehow/medqa")["test"]
    print(d)

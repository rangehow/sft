from vllm import SamplingParams

dname2samplingparams = {}


def register2dict(name):
    def decorator(func):
        dname2samplingparams[name] = func
        return func

    return decorator


@register2dict(name="gsm8k")
def gsm8k():
    return SamplingParams(
        max_tokens=512,
        temperature=0,
        stop=[
            "}\n",
            "<|endoftext|>",
            "<|im_end|>",
            "</s>",
            "## Question:",
            "<|eot_id|>",
            "\n\nQuestion:",
            "Q:",
        ],
    )


@register2dict(name="humaneval")
def humaneval():
    return SamplingParams(
        max_tokens=512,
        temperature=0,
        stop=[
            "\nclass",
            "\ndef",
            "\n#",
            "\n@",
            "\nprint",
            "\nif",
            "\n```",
            "<file_sep>",
        ],
    )


@register2dict(name="mmlu")
def mmlu():
    return SamplingParams(
        max_tokens=512,
        temperature=0,
        stop=[
            "}\n",
            "<|endoftext|>",
            "<|im_end|>",
            "</s>",
            "## Question:",
            "<|eot_id|>",
            "\n\nQuestion:",
        ],
    )


@register2dict(name="medical")
def medical():
    return SamplingParams(
        max_tokens=2,
        temperature=0,
        stop=[
            "}\n",
            "<|endoftext|>",
            "<|im_end|>",
            "</s>",
            "## Question:",
            "<|eot_id|>",
            "\n\nQuestion:",
        ],
    )


@register2dict(name="medqa")
def medqa():
    return SamplingParams(
        max_tokens=512,
        temperature=0,
        stop=[
            "}\n",
            "<|endoftext|>",
            "<|im_end|>",
            "</s>",
            "## Question:",
            "<|eot_id|>",
            "\n\nQuestion:",
            "(A)",  # 防止无脑把答案都生成一遍……
            "(B)",
            "(C)",
            "(D)",
            "(E)",
        ],
        include_stop_str_in_output=True,  # 给模糊匹配一线生机
        min_tokens=1,  # 防止逆天空生成
        # frequency_penalty=0.2,
    )


# @register2dict(name="medmcqa")
# def medmcqa():
#     return SamplingParams(
#         max_tokens=1024,
#         temperature=0,
#         stop=["\n}"],
#         frequency_penalty=0.2,
#     )

dname2samplingparams["medmcqa"] = medqa
dname2samplingparams["pubmedqa"] = medqa


@register2dict(name="bbh")
def bbh():
    return SamplingParams(
        max_tokens=1024,
        temperature=0,
        stop=[
            "}\n",
            "<|endoftext|>",
            "<|im_end|>",
            "</s>",
            "## Question:",
            "<|eot_id|>",
            "\n\nQuestion:",
            "Q:",
        ],
    )


@register2dict(name="truthfulqa")
def truthfulqa():
    return SamplingParams(temperature=0, prompt_logprobs=0, max_tokens=512)


if __name__ == "__main__":
    d = datasets.load_dataset("cais/mmlu")
    print(d)

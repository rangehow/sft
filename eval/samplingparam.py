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
        stop=["Q:"],
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
        stop=["\n\nQuestion:"],
    )


@register2dict(name="bbh")
def bbh():
    return SamplingParams(
        max_tokens=1024,
        temperature=0,
        stop=["Q:"],
    )

@register2dict(name="truthfulqa")
def truthfulqa():
    return SamplingParams(temperature=0, prompt_logprobs=0, max_tokens=1)


if __name__ == "__main__":
    d = datasets.load_dataset("cais/mmlu")
    print(d)

from vllm import  SamplingParams

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


@register2dict(name="mmlu")
def mmlu():
    return  SamplingParams(
                max_tokens=512,
                temperature=0,
                stop=["\n\nQuestion:"],
            )


if __name__ == "__main__":
    d=datasets.load_dataset("cais/mmlu")
    print(d)
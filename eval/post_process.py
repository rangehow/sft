# 为每个数据集定义一个接口？接口应该是 ref和prediction就行，然后里面负责后处理和调用metrics
import re

import numpy as np

dname2post = {}


def register2dict(name):
    def decorator(func):
        dname2post[name] = func
        return func

    return decorator


@register2dict(name="gsm8k")
def gsm8k(prediciton, reference, vllm):

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

    correct, correct1 = 0, 0

    for p, r in zip(prediciton, reference):

        if vllm:
            generated_text = p.outputs[0].text
        else:
            generated_text = p
        all_responses = generated_text.split("\nQ:")[0]
        short_responses = maybe_remove_comma(find_number(all_responses))

        # print(f"Short answer: {short_responses}")

        try:
            correct += float(maybe_remove_comma(find_number(r))) == float(
                short_responses
            )
        except:
            correct += maybe_remove_comma(find_number(r)) == maybe_remove_comma(
                find_number(short_responses)
            )

        # print("-" * 40)
        # print(f"generated answer {all_responses}")
        # print(f"Short ground truth answer {find_number(r)}")
        # print(f"correct {correct}")
        # # print(f"Correct: {correct} out of {idx+1}")
        # print("=" * 40)

    return correct / len(reference) * 100


@register2dict(name="mmlu")
def mmlu(prediciton, reference, vllm):

    idx2char = {0: "A", 1: "B", 2: "C", 3: "D"}
    correct = 0
    idx = 0
    for p, r in zip(prediciton, reference):
        if vllm:
            generated_text = p.outputs[0].text
        else:
            generated_text = p

        # print(f"generate_text:\n {generated_text}\n")
        all_responses = generated_text[1]
        # import pdb
        # pdb.set_trace()
        if all_responses == idx2char[r]:
            correct += 1

        # print("-" * 40)
        # print(f"generated answer:\n {all_responses}")
        # print(f"Short generated answer:{short_responses}")
        # print(f"ground truth answer: {idx2char[r]}")
        # print(f"Correct: {correct} out of {idx+1}")
        # print("=" * 40)
        idx += 1
    print(f"total_data_: {len(reference)}")
    return correct / len(reference) * 100


@register2dict(name="humaneval")
def humaneval(prediciton, reference, vllm):
    from human_eval.data import write_jsonl
    import os

    samples = [
        dict(task_id=task_id, completion=completion.outputs[0].text)
        for completion, task_id in zip(prediciton, reference)
    ]
    write_jsonl("samples.jsonl", samples)
    with os.popen("evaluate_functional_correctness samples.jsonl") as stream:
        output = stream.read()
    match = re.search(r"\{.*\}", output)

    if match:
        dict_str = match.group()
        result_dict = eval(dict_str)  
        return result_dict["pass@1"]*100
    else:
        print("No dictionary found")
    




@register2dict(name="bbh")
def bbh(prediciton, reference, vllm):

    
    correct=0
    for p, r in zip(prediciton, reference):

        
        generated_text = p.outputs[0].text

        all_responses = generated_text.split("Q:")[0].split('the answer is ')[-1].lower()
        
        if all_responses == r.lower():
            correct+=1
            
        print("-" * 40)
        print(f"generated answer {all_responses}")
        print(f"Short ground truth answer {r.lower()}")
        print(f"correct {correct}")
        # print(f"Correct: {correct} out of {idx+1}")
        print("=" * 40)
        
    return correct / len(reference) * 100


def _parse_logprobs(tokens: list, outputs, ctxlen: int) -> tuple[float, bool]:
    # The first entry of prompt_logprobs is None because the model has no previous tokens to condition on.
    continuation_logprobs_dicts = outputs[0].prompt_logprobs

    def coerce_logprob_to_num(logprob):
        return getattr(logprob, "logprob", logprob)

    continuation_logprobs_dicts = [
        (
            {
                token: coerce_logprob_to_num(logprob)
                for token, logprob in logprob_dict.items()
            }
            if logprob_dict is not None
            else None
        )
        for logprob_dict in continuation_logprobs_dicts
    ]

    # Calculate continuation_logprobs
    # assume ctxlen always >= 1
    continuation_logprobs = sum(
        logprob_dict.get(token)
        for token, logprob_dict in zip(
            tokens[ctxlen:], continuation_logprobs_dicts[ctxlen:]
        )
    )
    return continuation_logprobs


@register2dict(name="truthfulqa")
def truthfulqa(prediciton, reference, vllm):

    def process_results_mc2(doc, results):
        acc, num = 0, 0  # 准确率，问题数
        # 计算模型对于每个问题的准确率
        for labels, ll in zip(doc, results):
            # print("\n\n\nlabels=",labels)
            # print("\ll=",ll)
            split_idx = labels.index(0)
            ll_true, ll_false = ll[:split_idx], ll[split_idx:]
            p_true, p_false = np.exp(np.array(ll_true)), np.exp(np.array(ll_false))
            p_true = p_true / (sum(p_true) + sum(p_false))

            acc += sum(p_true)
            num += 1

        return acc / num

    results = []
    ctx_len = [t[-1] for t in reference]
    for pre, len in zip(prediciton, ctx_len):
        res = []
        for p in pre:
            prompt_log = _parse_logprobs(
                tokens=p[0].prompt_token_ids, outputs=p, ctxlen=len
            )
            res.append(prompt_log)
        results.append(res)
    return process_results_mc2([labels[:-1] for labels in reference], results)

# 为每个数据集定义一个接口？接口应该是 ref和prediction就行，然后里面负责后处理和调用metrics
import re

import numpy as np
from .json_utils import *

dname2post = {}


def register2dict(name):
    def decorator(func):
        dname2post[name] = func
        return func

    return decorator


@register2dict(name="gsm8k")
def gsm8k(prediciton, reference):

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
        generated_text = p.outputs[0].text

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
def mmlu(prediciton, reference):

    idx2char = {0: "A", 1: "B", 2: "C", 3: "D"}
    correct = 0
    idx = 0
    for p, r in zip(prediciton, reference):

        generated_text = p.outputs[0].text

        # print(f"generate_text:\n {generated_text}\n")
        all_responses = generated_text[1]
        # import pdb
        # pdb.set_trace()
        if all_responses == idx2char[r]:
            correct += 1
        # else:
        #     print("-" * 40)
        #     print(f"generated answer:\n{generated_text}")
        #     print(f"Short generated answer:{all_responses}")
        #     print(f"ground truth answer: {idx2char[r]}")
        #     print(f"Correct: {correct} out of {idx+1}")
        #     print("=" * 40)
        # idx += 1
    print(f"total_data_: {len(reference)}")
    return correct / len(reference) * 100


@register2dict(name="medqa")
def medqa(prediciton, reference):
    solved_examples = 0
    num_total_examples = len(reference)
    no_answer = 0
    ambigious = 0
    for p, r in zip(prediciton, reference):
        # Read and Parse the prediction from model output

        prediction_str = p.outputs[0].text
        prediction_json = extract_first_complete_json(prediction_str)
        if prediction_json is None or "answer" not in prediction_json:
            prediction_json = extract_values_from_json(
                prediction_str, allow_no_quotes=True
            )
        if (
            prediction_json is None
            or "answer" not in prediction_json
            or prediction_json["answer"] is None
            or prediction_json["answer"] == ""
        ):
            # try_extracted_answer = model_specific_extraction(model, prediction_str)
            # if try_extracted_answer:
            #     # print(f"Extracted answer from model: {try_extracted_answer}")
            #     prediction_json["answer"] = try_extracted_answer
            # else:
            if f"({r})" in prediction_str:
                ambigious += 1
                print("模糊命中")
                print(prediction_str)
                print("答案: ", r)
                print("=" * 40)
                solved_examples += 1
            else:
                no_answer += 1
                # print the no answer examples for debugging
                # if False and "Llama-3.1" in model:
                #     print(f"No answer for {item['id']}")
                # print("-" * 40)
                # print("解析失败")
                # print(prediction_str)
                # print(prediction_json)
                # print("=" * 40)
            continue
        reason = prediction_json.get("reasoning", "")
        model_answer = prediction_json["answer"]
        if model_answer == r or f"{r})" in model_answer:
            solved_examples += 1

        else:
            continue
            # print("-" * 40)
            # print(f"answer: {model_answer}")
            # print(f"reason: {reason}")
            # print(f"ground truth answer: {r}")
            # print("=" * 40)

    print(f"解析失败比例：{no_answer/num_total_examples}")
    print(f"模糊命中比例：{ambigious/num_total_examples}")
    return solved_examples / num_total_examples * 100


@register2dict(name="medical")
def medical(prediciton, reference):

    correct = 0
    idx = 0
    for p, r in zip(prediciton, reference):

        generated_text = p.outputs[0].text

        # print(f"generate_text:\n {generated_text}\n")
        try:
            all_responses = generated_text[1]
            # import pdb
            # pdb.set_trace()
            if all_responses.lower() == r.lower():
                correct += 1
            else:
                print("-" * 40)
                print(f"generated answer:{generated_text}")
                print(f"ground truth answer: {r}")
                print(f"Correct: {correct} out of {idx+1}")
                print("=" * 40)
            idx += 1
        except Exception as e:
            print(e)
    return correct / len(reference) * 100


dname2post["pubmedqa"] = medqa
dname2post["medmcqa"] = medqa
dname2post["bioasq"] = medqa
dname2post["multimedqa"] = medqa
dname2post["careqa"] = medqa
# @register2dict(name="medmcqa")
# def medmcqa(prediciton, reference):

#     correct = 0
#     idx = 0
#     parse_fail_cnt = 0
#     for p, r in zip(prediciton, reference):

#         generated_text = p.outputs[0].text

#         # print(f"generate_text:\n {generated_text}\n")
#         try:
#             try:
#                 all_responses = eval(generated_text+'}')
#             except:
#                 try:
#                     all_responses = eval(
#                         '''{\"reasoning\": \"'''
#                         + generated_text.replace("\n", "")
#                         + """}"""
#                     )
#                 except:
#                     all_responses = eval(
#                         """{""" + generated_text.replace("\n", "") + """}"""
#                     )

#             # import pdb
#             # pdb.set_trace()
#             if all_responses["answer"].lower() == r.lower():
#                 correct += 1
#             else:
#                 print("-" * 40)
#                 print(f"generated answer:\n{generated_text}")
#                 print(f"ground truth answer: {r}")
#                 print(f"Correct: {correct} out of {idx+1}")
#                 print("=" * 40)
#         except:

#             parse_fail_cnt += 1
#         idx += 1
#     print("解析失败的计数", parse_fail_cnt, parse_fail_cnt / len(reference))
#     return correct / len(reference) * 100


@register2dict(name="humaneval")
def humaneval(prediciton, reference):
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
        return result_dict["pass@1"] * 100
    else:
        print("No dictionary found")


@register2dict(name="bbh")
def bbh(prediciton, reference):

    correct = 0
    for p, r in zip(prediciton, reference):

        generated_text = p.outputs[0].text

        # all_responses = generated_text.split("Q:")[0].split('the answer is ')[-1].strip().lower().replace('.','')
        all_responses = generated_text.split("Q:")[0].lower()
        pattern = r"(?<=the answer is )(.*?)(?=\.)"
        # 使用 re.search 进行匹配
        match = re.search(pattern, all_responses)
        # 检查是否匹配成功
        if match:
            # 提取匹配的内容
            result = match.group(1)

        if result == r.lower():
            correct += 1

        # else:
        #     print("-" * 40)
        #     print(f"origin answer: {generated_text}")
        #     print(f"generated answer: {all_responses}")
        #     print(f"Short ground truth answer: {r.lower()}")
        #     print(f"correct {correct}")
        #     # print(f"Correct: {correct} out of {idx+1}")
        #     print("=" * 40)

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
def truthfulqa(prediciton, reference):

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

# 为每个数据集定义一个接口？接口应该是 ref和prediction就行，然后里面负责后处理和调用metrics
import re

dname2post = {}


def register2dict(name):
    def decorator(func):
        dname2post[name] = func
        return func

    return decorator


@register2dict(name="gsm8k")
def gsm8k(prediciton, reference,vllm):

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
            generated_text=p
        all_responses = generated_text.split("\nQ:")[0]
        short_responses = maybe_remove_comma(find_number(all_responses))

        print(f"Short answer: {short_responses}")

        try:
            correct += float(maybe_remove_comma(find_number(r))) == float(
                short_responses
            )
        except:
            correct += maybe_remove_comma(find_number(r)) == maybe_remove_comma(
                find_number(short_responses)
            )

        print("-" * 40)
        print(f"generated answer {all_responses}")
        print(f"Short ground truth answer {find_number(r)}")
        print(f"correct {correct}")
        # print(f"Correct: {correct} out of {idx+1}")
        print("=" * 40)

    return correct / len(reference) * 100

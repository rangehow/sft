# 为每个数据集定义一个接口？接口应该是 ref和prediction就行，然后里面负责后处理和调用metrics
import re
dname2post = {}


def register2dict(name):
    def decorator(func):
        dname2post[name] = func
        return func

    return decorator


@register2dict(name='gsm8k')
def gsm8k(prediciton,reference):
    
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
        # Example: 5,600.00 -> 5600.00
        return x.replace(",", "")
    
    def maybe_remove_comma_test(x: str) -> str:
        # Example: 5,600.00 -> 5600.00
        return x.replace(",", "").rstrip("0").rstrip(".")
    
    for p, r in zip(prediciton,reference):
        
        generated_text=p.outputs[0].text 
        all_responses = generated_text.split("\nQ:")[0]
        short_responses = maybe_remove_comma(
            find_number(all_responses)
        )

        print(f"Short answer: {short_responses}")

        correct,correct1=0,0
        try:
            correct += float(
                maybe_remove_comma(find_number(r))
            ) == float(short_responses)
        except:
            correct += maybe_remove_comma(
                find_number(r)
            ) == maybe_remove_comma(find_number(short_responses))
        
        try:
            correct1 += float(
                maybe_remove_comma_test(find_number(r))
            ) == float(short_responses)
        except:
            correct1 += maybe_remove_comma_test(
                find_number(r)
            ) == maybe_remove_comma_test(find_number(short_responses))
        
        if correct!=correct1:
            import pdb
            pdb.set_trace()
        print("-" * 40)
        print(f"Ground truth answer {r}")
        print(
            f"Short ground truth answer {find_number(r)}"
        )
        # print(f"Correct: {correct} out of {idx+1}")
        print("=" * 40)
    print(correct,correct1)





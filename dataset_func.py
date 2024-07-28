"""
每个函数需要完成的事项是：
1. 区分训练/推断，两者的不同之处主要在于训练时的input_ids是带着labels的（decoder）模型，
2. 区分vllm/普通推断方式，vllm模型推断只需要提供list[str]，而普通（hf)推断则需要提供常规的dict(str:tensor)
3. 推断还需要区分是不是chat？
4. 考虑到测试时需要返回标准答案，所以test模式下的返回应该包含一个固定的key “answer”
"""

import json
from typing import Any
from loguru import logger

dname2func = {}


def register2dict():
    def decorator(func):
        if func.__name__ not in dname2func:
            dname2func[func.__name__] = func
        else:
            exit()
        return func

    return decorator


def reformate(i, o):

    if o is not None:
        chat_dict = [
            {
                "role": "user",
                "content": i,
            },  # 不要在这里加空格，因为他有role token隔开的
            {"role": "assistant", "content": o},
        ]
    else:
        chat_dict = ({"role": "user", "content": i},)

    return chat_dict


def _process(real_input, output, template, test=False, mode=0, pt=False, **kwargs):
    # 注意，尽管这里有kwargs，但不会有任何一个被真正使用，只是作为遗留行为的兼容参数

    
    input_ids, labels = [], []
    if pt:
        
        temp_case = template.tokenizer.encode("你好")
        add_bos_token = temp_case[0] == template.tokenizer.bos_token_id
        # print(add_bos_token, temp_case[0])
        for text in output:

            if add_bos_token:
                temp_label = template.tokenizer.encode(
                    template.tokenizer.bos_token + text ,
                    add_special_tokens=False,
                )+ [template.base_eos_token_id]
            else:
                temp_label = template.tokenizer.encode(
                    text,
                    add_special_tokens=False,
                )+ [template.base_eos_token_id]
            if len(temp_label)>8192 or len(temp_label)<20:
                continue
            labels.append(temp_label)
            
        return {"input_ids": labels, "labels": labels}
    else:
        for i, o in zip(real_input, output):
            if mode == 1:
                if test:
                    chat_dict = reformate(i, None)
                    input_id = template.tokenizer.apply_chat_template(
                        chat_dict, tokenize=False, add_generation_prompt=True
                    )
                    input_ids.append(input_id)
                else:
                    input_id, label = template.apply(reformate(i, o))
                    input_ids.append(input_id)
                    labels.append(label)
            else:
                if test:
                    return {"input_ids": real_input, "answer": output}
                else:
                    # 这个空格放在input_id还是label，我考虑的问题如下：
                    # 1.mono dataset的增强结合。应当是放在input_id合适，因为pt数据集的第一个词总不以空格开头。
                    # 2. 大模型在生成的时候，开头总是带着一些
                    exit()
                    input_id = template.tokenizer.encode(
                        i+" ",
                        add_special_tokens=False,
                    )
                    label = template.tokenizer.encode(
                        o + template.tokenizer.eos_token,
                        add_special_tokens=False,
                    )

                    input_ids.append(input_id + label)

                    labels.append([-100 for _ in range(len(input_id))] + label)

        if not test:
            return {"input_ids": input_ids, "labels": labels}
        else:
            # 返回dict是dataset map的要求，这倒是没办法。
            return {"input_ids": input_ids, "answer": output}


@register2dict()
def alpaca_cleaned(instances, template, test=False, mode=0):

    instruction, input, output = (
        instances["instruction"],
        instances["input"],
        instances["output"],
    )

    real_input = [ins + " " + inp for ins, inp in zip(instruction, input)]

    return _process(
        real_input=real_input, output=output, template=template, mode=mode, test=test
    )

    input_ids, labels = [], []

    for i, o in zip(real_input, output):
        if test:
            exit()
        input_id, label = template.apply(reformate(i, o))
        input_ids.append(input_id)
        labels.append(label)
    if not test:
        return {"input_ids": input_ids, "labels": labels}
    else:
        return {
            "input_ids": input_ids,
        }


dname2func["alpaca_gpt4"] = alpaca_cleaned


@register2dict()
def gsm8k(instances, shot=True, **kwargs):
    INSTURCTION = """As an expert problem solver solve step by step the following mathematical questions."""

    # The default gsm8k prompt from the CoT paper
    # https://arxiv.org/pdf/2201.11903.pdf page 35.
    SHOT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8."""

    PROMPT = """
Q: {question}
A:"""
    if shot:
        real_input = [
            INSTURCTION + SHOT + PROMPT.format(question=q)
            for q in instances["question"]
        ]
    else:
        real_input = [
            INSTURCTION + PROMPT.format(question=q) for q in instances["question"]
        ]
    output = instances["answer"]

    # return {"input_ids": real_input, "answer": output}

    return _process(real_input, output, **kwargs)


@register2dict()
def mmlu(instances, shot=True, **kwargs):

    PROMPT = """Question: {question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:"""
    # INSTRUCTION = "The following are multiple choice questions (with answers) about "

    shot_data_path = "sft/eval/mmlu_few_shot_promot.json"
    with open(shot_data_path, "r") as file:
        cot_prompts = json.load(file)

    real_input = []
    output = []

    # 遍历input_列表中的每个元素，并匹配cot_prompts中的前缀
    length = len(instances["question"])
    for i in range(length):

        question, answer, subject, choices = (
            instances["question"][i],
            instances["answer"][i],
            instances["subject"][i],
            instances["choices"][i],
        )

        if shot:
            # 如果shot为真，使用cot_prompts中的值
            real_input.append(
                cot_prompts[subject]
                + PROMPT.format(
                    question=question.strip(),
                    A=choices[0],
                    B=choices[1],
                    C=choices[2],
                    D=choices[3],
                )
            )

        else:
            raise RuntimeError("不允许MMLU-cot采用非shot模式")

        output.append(answer)

    return _process(real_input, output, **kwargs)


@register2dict()
def bbh(instances, shot=True, **kwargs):

    if not shot:
        logger.warning("bbh只支持shot，自动转成shot")

    real_input = instances["inputs"]
    target = instances["target"]

    return _process(real_input, target, **kwargs)


@register2dict()
def humaneval(instances, **kwargs):

    num_repeat = 1  # 因为是贪心算法，所以根本无所谓。
    task_id = [item for item in instances["task_id"] for _ in range(num_repeat)]
    real_input = [item for item in instances["prompt"] for _ in range(num_repeat)]

    return _process(real_input, task_id, **kwargs)


@register2dict()
def apps(
    instances,
    template,
    **kwargs,
):

    real_input, label = (
        instances["question"],
        instances["solutions"],
    )

    input_ids, labels = [], []

    for i, o in zip(real_input, label):

        input_ids.append(i)
        labels.append(eval(o)[0])

    return _process(input_ids, labels, template=template, **kwargs)


@register2dict()
def code(
    instances,
    template,
    **kwargs,
):
    real_input, label = (
        instances["query"],
        instances["answer"],
    )

    input_ids, labels = [], []

    for i, o in zip(real_input, label):

        input_ids.append(i)
        labels.append(o)

    return _process(input_ids, labels, template=template, **kwargs)


@register2dict()
def math(
    instances,
    template,
    **kwargs,
):

    real_input, label = (
        instances["problem"],
        instances["solution"],
    )

    return _process(real_input, label, template=template, **kwargs)


@register2dict()
def truthfulqa(instances, template, shot=True, mode=0, **kwargs):

    def tok_encode(input, tokenizer, add_special_tokens=True):
        input_ids = tokenizer.encode(input, add_special_tokens=add_special_tokens)
        return input_ids

    SHOT = "Q: What is human life expectancy in the United States?\nA: Human life expectancy in the United States is 78 years.\n\nQ: Who was president of the United States in 1955?\nA: Dwight D. Eisenhower was president of the United States in 1955.\n\nQ: Which party did he belong to?\nA: He belonged to the Republican Party.\n\nQ: What is the square root of banana?\nA: I have no comment.\n\nQ: How does a telescope work?\nA: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\nQ: Where were the 1992 Olympics held?\nA: The 1992 Olympics were held in Barcelona, Spain.\n\nQ: "
    res = []
    real_input = []
    output = []
    for question, mc2_target in zip(instances["question"], instances["mc2_targets"]):
        # SHOT+问题
        if shot:
            context = SHOT + question + "\nA: "
        else:
            context = question + "\nA: "
        labels = mc2_target["labels"]
        for ch in mc2_target["choices"]:
            # 问题+选项进行tokenizer

            inp = context + ch
            real_input.append(inp)
            output.append(mc2_target)
        # results.append(res)
    return {"input_ids": real_input, "answer": output}


@register2dict()
def magpie(instances, template, test=False, **kwargs):
    conversations = instances["conversations"]
    input_ids = []
    labels = []
    for conv in conversations:

        human, assistant = conv[0]["value"], conv[1]["value"]

        input_id, label = template.apply(reformate(human, assistant))
        input_ids.append(input_id)
        labels.append(label)
    if not test:
        return {"input_ids": input_ids, "labels": labels}
    else:
        return {"input_ids": input_ids}


@register2dict()
def redpajama(instances, template, test=False, **kwargs):
    labels = []
    for text in instances["text"]:
        text_id = template.tokenizer.encode(
            text + template.tokenizer.eos_token,
            add_special_tokens=False,
        )
        labels.append(text_id)
    return {"labels": labels}


@register2dict()
def test(instances, **kwargs):
    
    return _process(
        real_input=instances["text"],
        output=instances["text"],
        **kwargs
    )


@register2dict()
def wiki_medical(instances, template, test=False, mode=0):

    return _process(
        real_input=None,
        output=instances["page_text"],
        template=template,
        test=test,
        mode=mode,
        pt=True,
    )
    
@register2dict()
def pubmed(instances, template, test=False, mode=0):

    return _process(
        real_input=None,
        output=instances["text"],
        template=template,
        test=test,
        mode=mode,
        pt=True,
    )


@register2dict()
def medical_transcription(instances, template, test=False, mode=0):

    return _process(
        real_input=None,
        output=instances["text"],
        template=template,
        test=test,
        mode=mode,
        pt=True,
    )


@register2dict()
def textbooks(instances, template, test=False, mode=0):

    return _process(
        real_input=None,
        output=instances["content"],
        template=template,
        test=test,
        mode=mode,
        pt=True,
    )


@register2dict()
def medpt(instances, template, test=False, mode=0):

    output = []

    for q, a in zip(instances["questions"], instances["answers"]):
        q_str = q[0][0].replace("?", "？")
        a_str = a[0].replace(".", "。").replace("?", "？")
        if not q_str.endswith("？"):
            q_str += "？"

        output.append(q_str + " " + a_str)

    return _process(
        real_input=None, output=output, template=template, test=test, mode=mode, pt=True
    )


@register2dict()
def medquad(instances, template, test=False, mode=0):

    return _process(
        real_input=instances["Question"],
        output=instances["Answer"],
        template=template,
        test=test,
        mode=mode,
        pt=False,
    )


@register2dict()
def medmcqa(instances, **kwargs):

    PROMPT = """
## Instruction 
Answer question by reasoning first and then providing your answer.
Present your reasoning and solution in the following json format. 
Please show your final answer in the `answer` field, e.g.,`"answer": "C"`.

# Example
Question: Chronic urethral obstruction due to benign prismatic hyperplasia can lead to the following change in kidney parenchyma
A: Hyperplasia
B: Hyperophy
C: Atrophy
D: Dyplasia
{{
    "reasoning": "Chronic urethral obstruction because of urinary calculi, prostatic hyperophy, tumors, normal pregnancy, tumors, uterine prolapse or functional disorders cause hydronephrosis which by definition is used to describe dilatation of renal pelvis and calculus associated with progressive atrophy of the kidney due to obstruction to the outflow of urine.",
    "answer": "C"
}}

Question: Hyper viscosity is seen in
A: Cryoglobulinemia
B: Multiple myeloma
C: MGUS
D: Lymphoma
{{
    "reasoning": "The term cryoglobulinemia refers to the presence in the serum of proteins that precipitate at temperatures below 37 degrees C and redissolve on rewarming. ... The elective treatment for hyperviscosity syndrome, whether associated with monoclonal, mixed, or polyclonalcryoglobulinemia, is plasma exchange.",
    "answer": "A"
}}

# Real question
Question: {question}
A: {opa}
B: {opb}
C: {opc}
D: {opd}
"""
    cop2options = {0: "A", 1: "B", 2: "C", 3: "D"}
    # INSTRUCTION = "The following are multiple choice questions (with answers) about "

    real_input = []
    output = []

    for question, opa, opb, opc, opd, cop in zip(
        instances["question"],
        instances["opa"],
        instances["opb"],
        instances["opc"],
        instances["opd"],
        instances["cop"],
    ):

        real_input.append(
            PROMPT.format_map(
                {"question": question, "opa": opa, "opb": opb, "opc": opc, "opd": opd}
            )
        )
        output.append(cop2options[cop])

    return _process(real_input, output, **kwargs)

@register2dict()
def medqa(instances, **kwargs):

    PROMPT = """
    ## Question: 

    {question}

    ## Choices:

    {choices}

    ## Instruction 

    Please answer this question by first reasoning and then selecting the correct choice. 
    Present your reasoning and solution in the following json format. 
    Please show your choice in the `answer` field with only the choice letter, e.g.,`"answer": "C"`.

    ```json
    {{
        "reasoning": "___",
        "answer": "___"
    }}
    ```
    """
    real_input,answer=[],[]
    def generate_choice_string(choices):
        choice_string = ""
        for i,choice in enumerate(choices):
            choice_string += f"- ({chr(65 + i)}) {choice['value']}\n"
        return choice_string
    for instance in list(zip(*instances.values())):
        
        
        real_input.append(PROMPT.format_map({
            'question':instance[1],
            'choices':generate_choice_string(instance[4])
        }))
        answer.append(instance[2])
    return _process(real_input, output, **kwargs)
    






@register2dict()
def medical(instances, template, shot=False, test=False, mode=0):

    return _process(
        real_input=instances["question"],
        output=instances["answer"],
        template=template,
        test=test,
        mode=mode,
    )


if __name__ == "__main__":
    import datasets
    from template import modelType2Template
    from transformers import AutoTokenizer
    from functools import partial
    from loguru import logger
    from eval.load_func import dname2load
    dataset=dname2load['medqa'](None)
    tokenizer = AutoTokenizer.from_pretrained('google/gemma-2b')
    template=modelType2Template['gemma'](tokenizer)
    dataset = dataset.map(
            partial(
                dname2func['medqa'],
                template=template,
                mode=1,
                test=False,
            ),
            batched=True,
            num_proc=1,
            # remove_columns=train_dataset.features.keys(),
            load_from_cache_file=False,
            desc="tokenize",
        )
    for d in dataset['train']:
        
        input_ids = d["input_ids"]
        labels = d["labels"]

        if -100 in labels:
            filtered_tensor = labels[labels.index(-100) + labels.count(-100) :]
        else:
            filtered_tensor = labels
        logger.debug("input_ids")
        print(tokenizer.decode(input_ids))
        print(tokenizer.convert_ids_to_tokens(input_ids))
        logger.debug("labels")
        print(tokenizer.convert_ids_to_tokens(filtered_tensor))
        import pdb
        pdb.set_trace()
        

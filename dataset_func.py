"""
每个函数需要完成的事项是：
1. 区分训练/推断，两者的不同之处主要在于训练时的input_ids是带着labels的（decoder）模型，
2. 区分vllm/普通推断方式，vllm模型推断只需要提供list[str]，而普通（hf)推断则需要提供常规的dict(str:tensor)
3. 推断还需要区分是不是chat？
4. 考虑到测试时需要返回标准答案，所以test模式下的返回应该包含一个固定的key “answer”
"""
import json 
from typing import Any


dname2func = {}


def register2dict(name):
    def decorator(func):
        dname2func[name] = func
        return func

    return decorator


def reformate(i, o, template):

    if o is not None:
        chat_dict = [
            {"role": "user", "content": i},
            {"role": "assistant", "content": o},
        ]
    else:
        chat_dict = ({"role": "user", "content": i},)

    return chat_dict


def _process(real_input, output, template, test=False, vllm=False, chat=False, mode=0):

    input_ids, labels = [], []
    for i, o in zip(real_input, output):
        if mode == 1:
            if test:
                if vllm:
                    # vllm只需要自然文本输入，不需要自己tokenize，只需要封成template后的格式
                    chat_dict = reformate(i, None, template)
                    input_id = template.tokenizer.apply_chat_template(
                        chat_dict, tokenize=False, add_generation_prompt=True
                    )

                elif not vllm:
                    chat_dict = reformate(i, None, template)
                    input_id = template.tokenizer.apply_chat_template(
                        chat_dict, tokenize=True, add_generation_prompt=True
                    )
                input_ids.append(input_id)
            else:
                input_id, label = reformate(i, o, template)
                input_ids.append(input_id)
                labels.append(label)
        else:
            if test:
                
                return {"input_ids": real_input, "answer": output}
                    
            else:
                # 不允许base模式进入训练
                exit()

    if not test:
        return {"input_ids": input_ids, "labels": labels}
    else:
        # 返回dict是dataset map的要求，这倒是没办法。
        # print(input_ids)
        return {"input_ids": input_ids, "answer": output}


@register2dict(name="alpaca_cleaned")
def alpaca_cleaned(instances, template, test=False):

    instruction, input, output = (
        instances["instruction"],
        instances["input"],
        instances["output"],
    )

    real_input = [ins + inp for ins, inp in zip(instruction, input)]

    input_ids, labels = [], []

    for i, o in zip(real_input, output):
        if test:
            exit()
        input_id, label = template.apply(reformate(i, o, template))
        input_ids.append(input_id)
        labels.append(label)
    if not test:
        return {"input_ids": input_ids, "labels": labels}
    else:
        return {
            "input_ids": input_ids,
        }


dname2func["alpaca_gpt4"] = alpaca_cleaned


@register2dict(name="gsm8k")
def gsm8k(instances, shot=False, mode=0, **kwargs):
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

@register2dict(name="mmlu")
def mmlu(instances, shot=False, mode=0, **kwargs):
    
    PROMPT = """
Q: {question}\n(A) {A} (B) {B} (C) {C} (D) {D}\n
A: Let's think step by step."""

    # if shot:
        
    #     real_input = [
    #         SHOT + PROMPT.format(question=q,A=A,B=B,C=C,D=D)
    #         for q,A,B,C,D in zip(instances["input"],instances["A"],instances["B"],instances["C"],instances["D"])
    #     ]
    # else:
    #     real_input = [
    #         PROMPT.format(question=q,A=A,B=B,C=C,D=D)
    #         for q,A,B,C,D in zip(instances["input"],instances["A"],instances["B"],instances["C"],instances["D"])
    #     ]
    # output = instances["target"]

    # return {"input_ids": real_input, "answer": output}

    shot_data_path="../eval/a.json"
    # 读入mmlu所有分支的shot数据集, 如cot_prompts={"abstarct_algebra":"shot1","anamoy":"shot2"}
    with open(shot_data_path, 'r') as file:
        cot_prompts = json.load(file)
    real_input=[]
    output=[]
    t=0  
    if shot:
        
        """
        假设instances内容如下：
        instances={ "input":["abstarct_algebra","aaa","bbb"",anamoy":"ccc"],
                    "A":["abstarct_algebra","1","2","anamoy":"3"]
                    }
        则得到下面这样的real_input
        real_input=["shot1aaa1","shot1bbb2","shot2ccc3"]
        """
        
        for i in range(len(instances["input"])):
            sub=instances["input"][t]
            # 如果instances["input"][t]元素不等于cot_promts的键值，那么把该下标对应的值添加到real_input里
            if sub not in cot_prompts :

                real_input.append(
                        shot+PROMPT.format(
                        question=instances["input"][t],
                        A=instances["A"][t],
                        B=instances["B"][t],
                        C=instances["C"][t],
                        D=instances["D"][t])
                )
                output.append(instances["target"][t])
            # 如果instances["input"][t]元素值在cot_promts的关键词中
            else:
                #保存sub对应的shot值
                shot= cot_prompts[sub]
                #利用t+1来把instances对应的下标往后移
                t=t+1
                continue
            
            
    else:
         for i in range(len(instances["input"])):
            sub=instances["input"][t]
            if sub not in cot_prompts :

                real_input.append(
                        PROMPT.format(
                        question=instances["input"][t],
                        A=instances["A"][t],
                        B=instances["B"][t],
                        C=instances["C"][t],
                        D=instances["D"][t])
                )
                output.append(instances["target"][t])

            else:
                t=t+1
                continue

    return _process(real_input, output, **kwargs)

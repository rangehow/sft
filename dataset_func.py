"""
每个函数需要完成的事项是：
1. 区分训练/推断，两者的不同之处主要在于训练时的input_ids是带着labels的（decoder）模型，
2. 区分vllm/普通推断方式，vllm模型推断只需要提供list[str]，而普通（hf)推断则需要提供常规的dict(str:tensor)
3. 推断还需要区分是不是chat？
4. 考虑到测试时需要返回标准答案，所以test模式下的返回应该包含一个固定的key “answer”
"""

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
    INSTURCTION = """"""

    # The default gsm8k prompt from the CoT paper
    # https://arxiv.org/pdf/2201.11903.pdf page 35.
#     SHOT = """
# The following are multiple choice questions (with answers) about abstract\
#   \ algebra.\n\nQ: Statement 1 | Every element of a group generates a cyclic subgroup\
#   \ of the group. Statement 2 | The symmetric group S_10 has 10 elements.\n(A) True,\
#   \ True (B) False, False (C) True, False (D) False, True\nA: Let's think step by\
#   \ step. A cyclic group is a group that is generated by a single element. Hence a\
#   \ subgroup generated by a single element of a group is cyclic and Statement 1 is\
#   \ True. The answer is (C).\n\nQ: The symmetric group $S_n$ has $\nactorial{n}$ elements,\
#   \ hence it is not true that $S_{10}$ has 10 elements.\nFind the characteristic of\
#   \ the ring 2Z.\n(A) 0 (B) 3 (C) 12 (D) 30\nA: Let's think step by step. A characteristic\
#   \ of a ring is R is $n$ if the statement $ka = 0$ for all $a\\in 2Z$ implies that\
#   \ $k$ is a multiple of $n$. Assume that $ka = 0$ for all $a\\in 2Z$ for some $k$.\
#   \ In particular $2k = 0$. Hence $k=0$ and $n=0$. The answer is (A).\n\nQ: Statement\
#   \ 1| Every function from a finite set onto itself must be one to one. Statement\
#   \ 2 | Every subgroup of an abelian group is abelian.\n(A) True, True (B) False,\
#   \ False (C) True, False (D) False, True\nA: Let's think step by step. Statement\
#   \ 1 is true. Let $S$ be a finite set. If $f:S \nightarrow S$ is a onto function,\
#   \ then $|S| = |f(S)|$. If $f$ was not one to one, then for finite domain $S$ the\
#   \ image would have less than $S$ elements, a contradiction.\nStatement 2 is true.\
#   \ Let $G$ be an abelian group and $H$ be a subgroup of $G$. We need to show that\
#   \ $H$ is abelian. Let $a,b \\in H$. Then $a,b \\in G$ and $ab=ba$. Since $G$ is\
#   \ abelian, $ab=ba$. Since $H$ is a subgroup of $G$, $ab \\in H$. Therefore, $ab=ba$\
#   \ and $H$ is abelian. The answer is (A).\n\nQ: Statement 1 | If aH is an element\
#   \ of a factor group, then |aH| divides |a|. Statement 2 | If H and K are subgroups\
#   \ of G then HK is a subgroup of G.\n(A) True, True (B) False, False (C) True, False\
#   \ (D) False, True\nA: Let's think step by step. Statement 2 is false. Let $H$ be\
#   \ a subgroup of $S_3$ generated by the cycle $(1,2)$ and $K$ be a subgroup of $S_3$\
#   \ generated by the cycle $(1,3)$. Both $H$ and $K$ have two elements, the generators\
#   \ and the identity. However $HK$ contains cycles (1,2), (1,3) and (2,3,1), but the\
#   \ inverse of (2,3,1) is (2,1,3) and it does not belong to HK, hence HK is not a\
#   \ subgroup. The answer is (B).\n\nQ: Find all c in Z_3 such that Z_3[x]/(x^2 + c)\
#   \ is a field.\n(A) 0 (B) 1 (C) 2 (D) 3\nA: Let's think step by step. Z_3[x]/(x^2\
#   \ + c) is a field if and only if x^2 + c does not have roots in Z_3. That is x^2\
#   \ + c != 0 for every x in Z_3. If c = 0, then x^2 + c = x^2 has root 0. If c = 1\
#   \ then x^2 + c = x^2 + 1 = 0 + 1 for x = 0, 1 + 1 = 2 for x = 1 and 1 + 1 = 2 for\
#   \ x = 2, hence x^2 + 1 does not have any roots. For c = 2 the polynomial x^2 + 2\
#   \ has two roots at x = 1 and x = 2. Hence Z_3[x]/(x^2 + c) is a field if and only\
#   \ if c = 1. The answer is (B).\n\n
#     """

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
    import json 
    shot_data_path="/data/abudukeyumu/sft/eval/mmlu_cot_prompts.json"
    with open(shot_data_path, 'r') as file:
        cot_prompts = json.load(file)
    real_input=[]
    output=[]
    t=0
    if shot:
        for i in range(len(instances["input"])):
            sub=instances["input"][t]
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

            else:
                shot= cot_prompts[sub]
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

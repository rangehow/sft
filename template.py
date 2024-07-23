from transformers import GemmaTokenizer
from loguru import logger

modelType2Template = {}


def register_template(cls):
    modelType2Template[cls.model_type] = cls
    return


class Template:
    def __init__(
        self,
        tokenizer,
        user_token=None,
        assistant_token=None,
        start_token_id=None,
        end_token_id=None,
        system_token=None,
        efficient_eos=False,
        default_system=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.user_token = user_token if user_token else []
        self.assistant_token = assistant_token if assistant_token else []
        self.system_token = system_token if system_token else []
        self.efficient_eos = efficient_eos
        self.start_token_id = start_token_id if start_token_id else None
        self.end_token_id = end_token_id if end_token_id else None
        self.default_system = default_system if default_system else None

    def apply(self, messages: list[dict[str, str]]):

        if self.start_token_id:
            input_id = [self.start_token_id]
            label = [-100]
        else:
            input_id, label = [], []

        start_idx = 0
        first_user_flag_for_efficient_eos = True
        if messages[0]["role"] == "system":
            exit()  # 还没实现
        elif self.default_system:
            system_token = self.tokenizer.encode(
                self.system_token.format_map({"content": self.default_system}),
                add_special_tokens=False,
            )
            input_id += system_token
            label += [-100] * len(system_token)
        for i in range(start_idx, len(messages)):

            if messages[i]["role"] == "user":
                if i == 0:
                    user_token = self.tokenizer.encode(
                        self.user_token.format_map({"content": messages[i]["content"]}),
                        add_special_tokens=False,
                    )
                else:
                    user_token = self.tokenizer.encode(
                        self.user_token.format_map({"content": messages[i]["content"]}),
                        add_special_tokens=False,
                    )
                input_id += user_token
                if self.efficient_eos and not first_user_flag_for_efficient_eos:
                    label += [self.tokenizer.eos_token_id] + [-100] * (
                        len(user_token) - 1
                    )
                else:
                    first_user_flag_for_efficient_eos = False
                    label += [-100] * len(user_token)
            elif messages[i]["role"] == "assistant":

                assistant_token = self.tokenizer.encode(
                    self.assistant_token.format_map(
                        {"content": messages[i]["content"]}
                    ),
                    add_special_tokens=False,
                )

                input_id += assistant_token
                label += assistant_token
            else:
                error_role = messages[i]["role"]
                logger.error(f"未经定义的template类型{error_role}")

            # print(input_id)
            # print(label)
            # import pdb
            # pdb.set_trace()

        if self.efficient_eos:
            if self.end_token_id:
                input_id += [self.end_token_id]
                label += [self.end_token_id]
        return input_id, label


@register_template
class GemmaTemplate(Template):
    model_type = "gemma"

    def __init__(self, tokenizer) -> None:

        super().__init__(
            tokenizer=tokenizer,
            user_token="<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n",
            assistant_token="{content}<eos>\n",
            start_token_id=tokenizer.bos_token_id,
            # end_token_id=tokenizer.eos_token_id,
            efficient_eos=True,
        )


@register_template
class Qwen2Template(Template):
    model_type = "qwen2"

    def __init__(self, tokenizer) -> None:

        super().__init__(
            tokenizer=tokenizer,
            user_token="<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n",
            assistant_token="{content}<|im_end|>\n",
            start_token_id=tokenizer.bos_token_id,
            end_token_id=tokenizer.eos_token_id,
            efficient_eos=False,
            system_token="<|im_start|>system\n{content}<|im_end|>\n",
            default_system="You are a helpful assistant.",
        )


@register_template
class LlamaTemplate(Template):
    model_type = "llama"

    def __init__(self, tokenizer) -> None:

        super().__init__(
            tokenizer=tokenizer,
            user_token="<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            assistant_token="{content}<|eot_id|>",
            start_token_id=tokenizer.bos_token_id,
            end_token_id=tokenizer.eos_token_id,
            efficient_eos=False,
        )


@register_template
class Llama2Template(Template):
    model_type = "llama2"

    def __init__(self, tokenizer) -> None:

        super().__init__(
            tokenizer=tokenizer,
            user_token="<s>[INST] {content} [/INST]",
            assistant_token="{content} </s>",
            start_token_id=None,
            end_token_id=tokenizer.eos_token_id,
            efficient_eos=False,
        )


@register_template
class YiTemplate(Template):
    model_type = "yi"

    def __init__(self, tokenizer) -> None:

        super().__init__(
            tokenizer=tokenizer,
            user_token="<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n",
            assistant_token="{content}<|im_end|>\n",
            start_token_id=None,
            end_token_id=None,  # yi的结束词就是<|im_end|>
            efficient_eos=False,
            system_token="<|im_start|>system\n{content}<|im_end|>\n",
            default_system=None,
        )


@register_template
class MistralTemplate(Template):
    model_type = "mistral"

    def __init__(self, tokenizer) -> None:

        super().__init__(
            tokenizer=tokenizer,
            user_token="[INST] {content}[/INST]",
            assistant_token=" {content}</s>",
            start_token_id=tokenizer.bos_token_id,
            end_token_id=None,
            efficient_eos=True,
        )


def test(tokenizer_name, template):
    access_token = "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"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        token=access_token,
    )

    g = modelType2Template[template](tokenizer)
    message = [
        {"role": "user", "content": "aaa"},
        {"role": "assistant", "content": "sad"},
        {"role": "user", "content": "aaa"},
        {"role": "assistant", "content": "smile"},
    ]
    c = tokenizer.apply_chat_template(message, tokenize=True)
    d = tokenizer.apply_chat_template(message, tokenize=False)
    a, b = g.apply(message)

    # if c[-1] != tokenizer.eos_token_id:
    #     print(tokenizer_name, "这个tokenizer的template不以eos结尾")
    #     print("原始结果:\n", c)
    #     print("模板结果:\n", a)
    #     print("结尾：", tokenizer.decode(c[-1]), c[-1])
    #     print("?", tokenizer.decode(tokenizer.eos_token_id))
    #     print("EOS：", tokenizer.eos_token, tokenizer.eos_token_id)
    # print(tokenizer)
    if a != c:
        print("=" * 30)
        print(tokenizer_name)
        print("原始结果:\n", c)
        print("模板结果:\n", a)
        print("---------")
        print("原始结果:\n", tokenizer.convert_ids_to_tokens(c))
        print("模板结果:\n", tokenizer.convert_ids_to_tokens(a))
        print("---------")
        print("原始结果:\n", d)
        print("模板结果:\n", tokenizer.decode(a))

    # print("---------")
    # print([tokenizer.convert_ids_to_tokens(cc) for cc in c])
    # print([tokenizer.convert_ids_to_tokens(aa) for aa in a])
    # print("---------")
    # print(b)
    # print(list(zip(a, b)))
    # print([tokenizer.convert_ids_to_tokens(bb) for bb in b if bb != -100])
    return a == c


if __name__ == "__main__":
    test_list = [
        ("mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
        ("mistralai/Mistral-Nemo-Instruct-2407", "mistral"),
        
        # ("Qwen/Qwen1.5-32B-Chat", "qwen2"),
        # ("google/gemma-2-27b-it", "gemma"),
        # ("google/gemma-7b-it", "gemma"),
        # ("meta-llama/Meta-Llama-3-8B-Instruct", "llama"),
        # ("meta-llama/Llama-2-7b-chat-hf", "llama2"),
        # ("01-ai/Yi-1.5-34B-Chat", "yi"),
        # ("Qwen/Qwen2-72B-Instruct", "qwen2"),
    ]
    result = {}
    for instance in test_list:
        if not test(instance[0], instance[1]):
            result[instance[0]] = "fail"
        else:
            result[instance[0]] = "success"
    print(result)

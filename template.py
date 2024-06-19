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
        self.default_system= default_system if default_system else None
    def apply(self, messages: list[dict[str, str]]):
        
        
        if self.start_token_id:
            input_id = [self.start_token_id]
            label = [-100]
        else:
            input_id, label = [], []

        start_idx = 0
        first_user_flag_for_efficient_eos = True
        if messages[0]["role"] == "system":
            exit() # 还没实现
        elif  self.default_system:
            system_token = self.tokenizer.encode(
                    self.system_token.format_map({"content": self.default_system}),
                    add_special_tokens=False,
            )
            input_id+=system_token
            label += [-100] * len(system_token)
        for i in range(start_idx, len(messages)):
            
            if messages[i]["role"] == "user":
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
            assistant_token="{content}<end_of_turn>\n",
            start_token_id=tokenizer.bos_token_id,
            end_token_id=tokenizer.eos_token_id,
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
            default_system="You are a helpful assistant."
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


if __name__ == "__main__":
    access_token = "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"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2-1.5B-Instruct", token=access_token
    )
    print(tokenizer)
    g = modelType2Template["qwen2"](tokenizer)
    message = [
        {"role": "user", "content": "aaa"},
        {"role": "assistant", "content": "sad"},
        {"role": "user", "content": "aaa"},
        {"role": "assistant", "content": "smile"},
    ]
    c = tokenizer.apply_chat_template(message, tokenize=True)
    print(c)
    print('---------')
    a, b = g.apply(message)
    print(tokenizer.decode(a))
    print(b)
    print(list(zip(a,b)))
    print([tokenizer.convert_ids_to_tokens(bb) for bb in b if bb!=-100 ])
    assert a==c

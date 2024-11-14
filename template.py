from transformers import GemmaTokenizer
from loguru import logger

modelType2Template = {}


def register_template(cls):
    if isinstance(cls.model_type, list):
        for name in cls.model_type:
            modelType2Template[name] = cls
    else:
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
        tool_token=None,
        efficient_eos=False,
        default_system=None,
        jinja_template=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.user_token = user_token if user_token else []
        self.assistant_token = assistant_token if assistant_token else []
        self.system_token = system_token if system_token else []
        self.tool_token = tool_token if tool_token else []

        self.efficient_eos = efficient_eos
        self.start_token_id = start_token_id if start_token_id else None
        self.end_token_id = end_token_id if end_token_id else None
        self.default_system = default_system if default_system else None
        self.base_eos_token_id = tokenizer.eos_token_id
        self.chat_eos_token_id = tokenizer.eos_token_id

        self.jinja_template = None
        
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
            if self.model_type == "llama31" and self.tool_token != []:
                system_token = self.tokenizer.encode(
                    self.system_token.format_map(
                        {"content": "Environment: ipython\n" + self.default_system}
                    ),
                    add_special_tokens=False,
                )
            else:
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
    model_type = ["gemma", "gemma2"]

    def __init__(self, tokenizer) -> None:

        super().__init__(
            tokenizer=tokenizer,
            user_token="<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n",
            assistant_token="{content}<end_of_turn>\n",
            start_token_id=tokenizer.bos_token_id,
            # end_token_id=tokenizer.eos_token_id,
            efficient_eos=True,
        )
        self.base_eos_token_id = 1
        self.chat_eos_token_id = 107


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
        # 必须写在后面不然会被默认值覆盖
        self.base_eos_token_id = 151643
        self.chat_eos_token_id = 151645


@register_template
class Qwen25Template(Template):
    model_type = "qwen2.5"

    def __init__(self, tokenizer) -> None:
        
        super().__init__(
            tokenizer=tokenizer,
            user_token="<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n",
            assistant_token="{content}<|im_end|>\n",
            start_token_id=tokenizer.bos_token_id,
            end_token_id=tokenizer.eos_token_id,
            efficient_eos=False,
            system_token="<|im_start|>system\n{content}<|im_end|>\n",
            default_system="You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
            jinja_template="{%- if tools %}\n    {{- '<|im_start|>system\\n' }}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- messages[0]['content'] }}\n    {%- else %}\n        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}\n    {%- endif %}\n    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}\n    {%- for tool in tools %}\n        {{- \"\\n\" }}\n        {{- tool | tojson }}\n    {%- endfor %}\n    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}\n        {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}\n    {%- else %}\n        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}\n    {%- endif %}\n{%- endif %}\n{%- for message in messages %}\n    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) or (message.role == \"assistant\" and not message.tool_calls) %}\n        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n    {%- elif message.role == \"assistant\" %}\n        {{- '<|im_start|>' + message.role }}\n        {%- if message.content %}\n            {{- '\\n' + message.content }}\n        {%- endif %}\n        {%- for tool_call in message.tool_calls %}\n            {%- if tool_call.function is defined %}\n                {%- set tool_call = tool_call.function %}\n            {%- endif %}\n            {{- '\\n<tool_call>\\n{\"name\": \"' }}\n            {{- tool_call.name }}\n            {{- '\", \"arguments\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- '}\\n</tool_call>' }}\n        {%- endfor %}\n        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}\n        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}\n            {{- '<|im_start|>user' }}\n        {%- endif %}\n        {{- '\\n<tool_response>\\n' }}\n        {{- message.content }}\n        {{- '\\n</tool_response>' }}\n        {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}\n            {{- '<|im_end|>\\n' }}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n{%- endif %}\n"
        )
        # 必须写在后面不然会被默认值覆盖
        self.base_eos_token_id = 151643
        self.chat_eos_token_id = 151645



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
        self.base_eos_token_id = 128001
        self.chat_eos_token_id = 128009


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
class Llama31Template(Template):
    model_type = "llama31"

    def __init__(self, tokenizer) -> None:

        super().__init__(
            tokenizer=tokenizer,
            user_token="<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            assistant_token="{content}<|eot_id|>",
            start_token_id=tokenizer.bos_token_id,
            end_token_id=tokenizer.eos_token_id,
            efficient_eos=False,
            system_token="<|start_header_id|>system<|end_header_id|>\n\n{content}\n\n<|eot_id|>",
            default_system="""Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024""",
            tool_token=None,
        )
        self.base_eos_token_id = 128001
        self.chat_eos_token_id = 128009


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


@register_template
class MistralNemoTemplate(Template):
    model_type = "mistral_nemo"

    def __init__(self, tokenizer) -> None:

        super().__init__(
            tokenizer=tokenizer,
            user_token="[INST]{content}[/INST]",
            assistant_token="{content}</s>",
            start_token_id=tokenizer.bos_token_id,
            end_token_id=None,
            efficient_eos=True,
        )


@register_template
class Phi3Template(Template):
    model_type = "phi3"

    def __init__(self, tokenizer) -> None:
        super().__init__(
            tokenizer=tokenizer,
            user_token="<|user|>\n{content}<|end|>\n<|assistant|>\n",
            assistant_token="{content}<|end|>\n",
            start_token_id=None,
            end_token_id=32000,
            efficient_eos=True,
        )


@register_template
class Phi3SamllTemplate(Template):
    model_type = "phi3small"

    def __init__(self, tokenizer) -> None:
        super().__init__(
            tokenizer=tokenizer,
            user_token="<|user|>\n{content}<|end|>\n<|assistant|>\n",
            assistant_token="{content}<|end|>\n",
            start_token_id=tokenizer.eos_token_id,
            end_token_id=tokenizer.eos_token_id,
            efficient_eos=True,
        )


# def test_tool_function():


def test(tokenizer_name, template):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

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
    
    if c[-1] != tokenizer.eos_token_id:
        print(tokenizer_name, "这个tokenizer的template不以eos结尾")

        # print("原始结果:\n", c)
        # print("模板结果:\n", a)
        # print("结尾：", tokenizer.decode(c[-1]), c[-1])
        # print("?", tokenizer.decode(tokenizer.eos_token_id))
        # print("EOS：", tokenizer.eos_token, tokenizer.eos_token_id)
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
        # ("mistralai/Mistral-7B-Instruct-v0.3", "mistral"),
        # ("mistralai/Mistral-Nemo-Instruct-2407", "mistral_nemo"),
        # ("microsoft/Phi-3-mini-4k-instruct", "phi3"),
        # ("Qwen/Qwen1.5-32B-Chat", "qwen2"),
        # ("microsoft/Phi-3-small-8k-instruct", "phi3small"),
        # ("microsoft/Phi-3-mini-4k-instruct", "phi3"),
        # ("google/gemma-7b-it", "gemma2"),
        # ("google/gemma-2-2b-it", "gemma2"),
        # ("/mnt/rangehow/models/Meta-Llama-3.1-8B-Instruct", "llama31"),
        # ("meta-llama/Llama-2-7b-chat-hf", "llama2"),
        # ("01-ai/Yi-1.5-34B-Chat", "yi"),
        # ("Qwen/Qwen2-72B-Instruct", "qwen2"),
        ("/mnt/rangehow/models/Qwen2.5-7B-Instruct", "qwen2.5"),
        
    ]
    result = {}
    for instance in test_list:
        if not test(instance[0], instance[1]):
            result[instance[0]] = "fail"
        else:
            result[instance[0]] = "success"
    print(result)

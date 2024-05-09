
from dataclasses import dataclass
from typing import Any, Optional



def split_list_by_value(lst, value):
    result = []
    temp = []
    for item in lst:
        if item == value:
            result.append(temp)
            temp = []
        else:
            temp.append(item)
    if temp:
        result.append(temp)
    return result



import json
import os

Class2ChatToken={}

def register_chat_token(cls):
    Class2ChatToken[cls.name] = cls()
    return cls

class MyTokenizer:
    def __init__(self,tokenizer) -> None:
        try:
            name_or_path=tokenizer.name_or_path
            config_dir=os.path.join(name_or_path,'tokenizer_config.json')
            if os.path.exists(config_dir):
                tokenizer_class=json.load(open(config_dir))['tokenizer_class']
                self.chat_token=Class2ChatToken[tokenizer_class]
            else:
                print('该路径不存在，可能不是本地导入的tokenizer，自行指定')
        except Exception as e:
            print(e)

        
        try:
            tokenizer._add_bos_token=False
            tokenizer._add_eos_token=False
        except Exception as e:
            print(e)
        
        self.tokenizer=tokenizer
    
    def apply(self, role,text,*args: Any, **kwds: Any) -> Any:
        tokens = {
            'system': self.chat_token.system_tokens,
            'user': self.chat_token.user_tokens,
            'assistant': self.chat_token.assistant_tokens
        }
        text=tokens[role].format_map({'text':text})
        return self.tokenizer(text,add_special_tokens=False,return_attention_mask=False)


@dataclass
class ChatToken:
    system_tokens:Optional[str]
    user_tokens:str
    assistant_tokens:str

@register_chat_token
class GemmaChatToken(ChatToken):
    name='GemmaTokenizer'
    def __init__(self):
        
        super().__init__(
            system_tokens=None,
            user_tokens="<start_of_turn>user\n{text}<end_of_turn>\n",
            assistant_tokens="<start_of_turn>assistant\n{text}<end_of_turn>\n"
        )
    


from transformers import GemmaTokenizer
from loguru import logger

modelType2Template={}

def register_template(cls):
    modelType2Template[cls.model_type]=cls
    return 

class Template:
    def __init__(self,tokenizer,user_token=None,assistant_token=None,system_token=None,efficient_eos=False) -> None:
        self.tokenizer=tokenizer
        self.user_token= user_token if user_token else []
        self.assistant_token= assistant_token if assistant_token else []
        self.system_token= system_token if system_token else []
        self.efficient_eos=efficient_eos
        
    def apply(self,messages:list[dict[str,str]]):
        input_id=[]
        label=[]
        start_idx=0
        first_user_flag_for_efficient_eos=True
        if messages[0]['role']=='system':
            chat_tokens+=self.system_token+messages[0]['content']
            start_idx=1
        for i in range(start_idx,len(messages)):
            if messages[i]['role']=='user':
                user_token=self.tokenizer.encode(self.user_token.format_map({'content':messages[i]['content']}),add_special_tokens=False)
                input_id+=user_token
                if self.efficient_eos and not first_user_flag_for_efficient_eos:
                    label+=[self.tokenizer.eos_token_id]+[-100]*(len(user_token)-1)
                else:
                    first_user_flag_for_efficient_eos=False
                    label+=[-100]*len(user_token)
            elif messages[i]['role']=='assistant':
                assistant_token=self.tokenizer.encode(self.assistant_token.format_map({'content':messages[i]['content']}),add_special_tokens=False)
                input_id+=assistant_token
                label+=assistant_token
            else:
                error_role=messages[i]['role']
                logger.error(f'未经定义的template类型{error_role}')
        if self.efficient_eos:
            input_id+=[self.tokenizer.eos_token_id]
            label+=[self.tokenizer.eos_token_id]
        return input_id,label

@register_template
class GemmaTemplate(Template):
    model_type='gemma'
    def __init__(self,tokenizer) -> None:
        super().__init__(tokenizer=tokenizer,user_token='<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n', assistant_token='{content}<end_of_turn>\n', system_token=GemmaTokenizer.bos_token,efficient_eos=True)
        

 
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('/data/ruanjh/best_training_method/gemma-2b')
# g=GemmaTemplate(tokenizer)
# a,b=g.apply([
#     {'role':'user','content':'aaa'},
#     {'role':'assistant','content':'ffff'},
#     {'role':'user','content':'aaa'},
#     {'role':'assistant','content':'ffff'},
# ])
# print(tokenizer.decode(a))
# print(list(zip(a[:-1],b[1:])))
from datasets import load_dataset
import datasets
# dataset = load_dataset("LDJnr/Capybara")
from config import gemma_dir
from transformers import AutoTokenizer
# dataset.save_to_disk('/data/ruanjh/best_training_method/sft/Capybara')
from functools import partial
import torch
torch.set_printoptions(edgeitems=2000)



def capybara(x,tokenizer):
    def reformate(c):
        dialogue_list=[]
        for turn in c:
            dialogue_list.append(
                {'role':'user','content':turn['input']},
            )
            dialogue_list.append(
                {'role':'assistant','content':turn['output']},
            )
        return dialogue_list
    
    def gen_label_from_input_id(x):
        # 只能处理一个tensor
        i=0
        special_token_set={0,1,2,3,106,107}
        while i <len(x)-1:
            # 这是区间反转，还需要考虑单独落在外面的special token
            if x[i]==106 and x[i+1]==1645:
                j=i
                while x[j]!=107:
                    x[j]=-100
                    j+=1
            elif x[i]==106 and x[i+1]==2516: # <startofturn>model \n
                x[i:i+3]=-100
                i+=3
            elif x[i]==107 and x[i+1]==108:# <end of text>\n
                x[i:i+2]=-100
                i+=2
            elif x[i].item() in special_token_set:
                x[i]=-100
                i+=1
            else:
                i+=1
        return x

        
    
    input_ids=[]
    labels=[]
    for c in x['conversation']:
        c=reformate(c)
        c=tokenizer.apply_chat_template(c,tokenize=True,return_tensors='pt')
        input_ids.append(c[0])
        label=gen_label_from_input_id(c[0].clone())
        labels.append(label)
        
    return {'input_ids':input_ids,'labels':labels}




dataset=datasets.load_from_disk('/data/ruanjh/best_training_method/sft/Capybara')['train']
model_dir=gemma_dir
tokenizer=AutoTokenizer.from_pretrained(model_dir)

dataset=dataset.map(partial(capybara,tokenizer=tokenizer),batched=True,num_proc =1,remove_columns=['source','conversation'],load_from_cache_file =True,cache_file_name='/data/ruanjh/best_training_method/sft/Capybara/cache.arrow')


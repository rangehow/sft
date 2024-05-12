
dname2func={}

def register2dict(func):
    dname2func[func.__name__]=func
    return func

def reformate(i, o,template):
    
    if o is not None:
        chat_dict = [
            {"role": "user", "content": i},
            {"role": "assistant", "content": o},
        ]
        return template.apply(chat_dict)
    else:
        chat_dict = [
            {"role": "user", "content": i},
        ]
        return template.apply(chat_dict)
    

@register2dict
def alpaca_cleaned(instances, template):
    instruction, input, output = (
        instances["instruction"],
        instances["input"],
        instances["output"],
    )
    real_input = [ins + inp for ins, inp in zip(instruction, input)]

    input_ids,labels=[],[]
    
    for i,o in zip(real_input,output):
        input_id,label=reformate(i,o,template)
        input_ids.append(input_id)
        labels.append(label)
    return {'input_ids':input_ids,'labels':labels}

dname2func['alpaca_gpt4']=alpaca_cleaned

def capybara(x,template):
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
            if x[i]==106 and x[i+1]==1645:  # <startofturn>user
                j=i
                while x[j]!=107:
                    x[j]=-100 
                    j+=1
            elif x[i]==106 and x[i+1]==2516: # <startofturn>model \n
                x[i:i+3]=-100
                i+=3
            elif x[i]==107 and x[i+1]==108:# <end of text>\n BUG eos是要学的
                # x[i+1]=-100 # 我觉得应该学\n，如果是多轮情况下需要在前序生成满足这个结构的话
                              # 如果是单轮倒无所谓，生成到EOS标志停止生成就行。
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
        for turn in c:
            tokenizer.apply(role='user',text=turn['input'])
            tokenizer.apply(role='assistant',text=turn['output'])
        c=reformate(c)
        c=tokenizer.apply_chat_template(c,tokenize=True)
        input_ids.append(c[0])
        label=gen_label_from_input_id(c[0].clone())
        labels.append(label)
        
    return {'input_ids':input_ids,'labels':labels}



    start_indices = (input_tensor[:-2] == 106) & (input_tensor[1:-1] == 1645)

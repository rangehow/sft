
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
        return chat_dict
    else:
        chat_dict = [
            {"role": "user", "content": i},
        ]
        return chat_dict
    

@register2dict
def alpaca_cleaned(instances, template,test=False):

    instruction, input, output = (
        instances["instruction"],
        instances["input"],
        instances["output"],
    )

    real_input = [ins + inp for ins, inp in zip(instruction, input)]

    input_ids,labels=[],[]
    
    for i,o in zip(real_input,output):
        if test:
            exit()
        input_id,label=template.apply(reformate(i,o,template))
        input_ids.append(input_id)
        labels.append(label)
    if not test:
        return {'input_ids':input_ids,'labels':labels}
    else:
        return {'input_ids':input_ids,}

dname2func['alpaca_gpt4']=alpaca_cleaned


@register2dict
def gsm8k(instances,template,test=False):
    real_input, output = (
        instances["question"],
        instances["answer"],
    )
    

    for i,o in zip(real_input,output):
        if test:
            chat_dict=reformate(i,None,template)
            text=template.tokenizer.apply_chat_template(chat_dict)
        else:
            input_id,label=reformate(i,o,template)
        input_ids.append(input_id)
        labels.append(label)
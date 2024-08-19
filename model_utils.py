import accelerate
from transformers import AutoModelForCausalLM


def balanced_load(model_dir, num_devices, is_distillation=False,ratio=None,devices_idx=None):
    assert sum(ratio)==1 or ratio==None,"ratio should be a list whose sum==1"
    assert len(devices_idx)==num_devices,"len(index of allocated device) should equal to num_devices "
    assert len(ratio)==num_devices,"len(ratio) should equal to num_devices "
    from collections import OrderedDict

    with accelerate.init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype="auto",
            trust_remote_code=True,
        )

    devices_idx=list(range(num_devices)) if devices_idx==None else devices_idx
    
    def create_manual_device_map(model, num_devices, is_distillation=False,ratio=ratio,devices_idx=devices_idx):
        
        num_layers = model.config.num_hidden_layers
        layers = [f"model.layers.{i}" for i in range(num_layers)]  # 假设有28层
        if ratio is not None:
            layers_per_device = [round(r * num_layers) for r in ratio]
        else:
            layers_per_device = [len(layers) // num_devices for _ in range(num_devices)]
            
        remainder = num_layers-sum(layers_per_device)
        
        # 把多出来的剩余层按从后向前的顺序分配一下
        for i in range(len(layers_per_device)-1,-1,-1):
            if remainder==0:
                break
            layers_per_device[i]+=1
            remainder-=1
        print(layers_per_device)
        if num_devices>=2:
            # 分摊一点给第二张卡
            layers_per_device[0]-=6 if is_distillation else 2
            layers_per_device[1]+=6 if is_distillation else 2
        
        device_map = OrderedDict()
        current_device = 0

        # 分配层到设备
        for layer in layers:
            while layers_per_device[current_device] ==0 :
                current_device+=1
            
            layers_per_device[current_device] -= 1
            device_map[layer] = devices_idx[current_device]


        # 分配其他模块
        device_map["model.embed_tokens"] = devices_idx[0]
        device_map["lm_head"] = devices_idx[0]
        device_map["model.norm"] = (
            devices_idx[num_devices - 1] if num_devices <= 2 else devices_idx[num_devices - 2]
        )

    
        return device_map

    # 使用手动创建的device_map
    device_map = create_manual_device_map(model, num_devices,is_distillation)

    # device_map["model.embed_tokens"] = device_map["lm_head"]
    # 打印device_map结果
    for module, device in device_map.items():
        print(f"{module}: {device}")

    del model

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map=device_map,
        attn_implementation="eager" if "gemma" in model_dir.lower() or 'phi' in model_dir.lower() else "sdpa",
        trust_remote_code=True,
    )
    return model


if __name__=='__main__':
    balanced_load('/mnt/rangehow/models/Meta-Llama-3.1-8B-Instruct',3,is_distillation=False,ratio=[0.2,0.4,0.4],devices_idx=[0,3,4])
import accelerate
from transformers import AutoModelForCausalLM


def balanced_load(model_dir, num_devices):
    from collections import OrderedDict

    with accelerate.init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype="auto",
            trust_remote_code=True,
        )

    def create_manual_device_map(model, num_devices):

        num_layers = model.config.num_hidden_layers
        layers = [f"model.layers.{i}" for i in range(num_layers)]  # 假设有28层
        layers_per_device = len(layers) // num_devices
        remainder = len(layers) % num_devices

        device_map = OrderedDict()
        current_device = 0
        current_layer = 0

        # 分配层到设备
        for layer in layers:
            device_map[layer] = current_device
            current_layer += 1
            if current_layer >= layers_per_device + (1 if remainder > 0 else 0):
                current_device += 1
                current_layer = 0
                remainder -= 1

        

        # 分配其他模块
        device_map["model.embed_tokens"] = 0
        device_map["lm_head"] = 0
        device_map["model.norm"] = (
            num_devices - 1 if num_devices <= 2 else num_devices - 2
        )
        # 因为
        max_layer_with_zero = max(
            (layer for layer, device in device_map.items() if device == 0 and "model.layers." in layer),
            key=lambda x: int(x.split(".")[-1])
        )

        # 将其 value 改为 1
        device_map[max_layer_with_zero] = 1
        
        
        return device_map

    # 使用手动创建的device_map
    device_map = create_manual_device_map(model, num_devices)

    # device_map["model.embed_tokens"] = device_map["lm_head"]
    # 打印device_map结果
    for module, device in device_map.items():
        print(f"{module}: {device}")

    # accelerate.load_checkpoint_in_model(model,student_model_dir,device_map=device_map)
    del model

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map=device_map,
        attn_implementation="eager" if "gemma" in model_dir.lower() or 'phi' in model_dir.lower() else "sdpa",
        trust_remote_code=True,
    )
    return model

import accelerate
from transformers import AutoModelForCausalLM
from collections import OrderedDict

def balanced_load(model_dir, num_devices):
    with accelerate.init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype="auto",
        )

    def create_manual_device_map(model, num_devices):
        # Calculate number of parameters for each layer
        num_layers = model.config.num_hidden_layers
        layer_weights = [
            model.get_submodule(f"transformer.h.{i}").num_parameters()
            for i in range(num_layers)
        ]
        
        # Calculate weights for other components
        embed_tokens_weight = model.get_submodule("transformer.wte").num_parameters()
        lm_head_weight = model.get_submodule("lm_head").num_parameters()
        norm_weight = model.get_submodule("transformer.ln_f").num_parameters()

        # Total parameters
        total_parameters = sum(layer_weights) + embed_tokens_weight + lm_head_weight + norm_weight
        params_per_device = total_parameters // num_devices

        device_map = OrderedDict()
        current_device = 0
        current_params = 0

        # Distribute layers
        for i, weight in enumerate(layer_weights):
            if current_params + weight > params_per_device and current_device < num_devices - 1:
                current_device += 1
                current_params = 0
            device_map[f"transformer.h.{i}"] = current_device
            current_params += weight

        # Assign other components
        device_map["transformer.wte"] = 0
        device_map["lm_head"] = 0
        device_map["transformer.ln_f"] = num_devices - 1

        return device_map

    device_map = create_manual_device_map(model, num_devices)

    for module, device in device_map.items():
        print(f"{module}: {device}")

    del model

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map=device_map,
        attn_implementation="eager" if "gemma" in model_dir else "sdpa",
    )
    if "gemma" in model_dir:
        print('using eager attention since gemma model')
    return model
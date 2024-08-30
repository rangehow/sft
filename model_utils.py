import accelerate
from transformers import AutoModelForCausalLM
import math
from torch.cuda import device_count


def balanced_load(
    model_dir,
    num_devices=device_count(),
    is_distillation=False,  # student 模型才需要写true，教师模型也false
    ratio=None,
    devices_idx=None,
):

    if ratio is not None:
        assert len(ratio) == num_devices, "len(ratio) should equal to num_devices"
        if sum(ratio) != 1:
            ratio = [d / sum(ratio) for d in ratio]

    from collections import OrderedDict
    import math
    from transformers import AutoModelForCausalLM
    import accelerate

    with accelerate.init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype="auto",
            trust_remote_code=True,
        )

    devices_idx = list(range(num_devices)) if devices_idx is None else devices_idx
    assert (
        len(devices_idx) == num_devices
    ), "len(index of allocated device) should equal to num_devices"

    def create_manual_device_map(
        model, num_devices, is_distillation=False, ratio=ratio, devices_idx=devices_idx
    ):
        device_map = {}
        current_device = 0

        # 计算每个模块的参数量
        lm_head_params = sum(p.numel() for p in model.lm_head.parameters())
        norm_params = sum(p.numel() for p in model.model.norm.parameters())
        rotary_emb_params = (
            sum(p.numel() for p in model.model.rotary_emb.parameters())
            if hasattr(model.model, "rotary_emb")
            else 0
        )
        layer_params = sum(p.numel() for p in model.model.layers[0].parameters())

        # 计算每个模块等效的层数
        ratio_lm_head = math.ceil(lm_head_params / layer_params)
        ratio_norm = math.ceil(norm_params / layer_params)
        ratio_rotary_emb = (
            math.ceil(rotary_emb_params / layer_params) if rotary_emb_params > 0 else 0
        )

        num_layers = model.config.num_hidden_layers
        total_layers = num_layers + ratio_lm_head + ratio_norm + ratio_rotary_emb

        # 确定每个设备应该分配到的层数
        if ratio is not None:
            layers_per_device = [round(r * total_layers) for r in ratio]
        else:
            layers_per_device = [
                total_layers // num_devices for _ in range(num_devices)
            ]

        remainder = total_layers - sum(layers_per_device)
        # 从后面开始分配剩余层
        for i in range(remainder - 1, -1, -1):
            layers_per_device[i] += 1
        layers = [f"model.layers.{i}" for i in range(num_layers)]

        # 将 lm_head、norm 和 rotary_emb 模块视为“层”进行分配
        special_layers = {
            "lm_head": ratio_lm_head,
            "model.norm": ratio_norm,
            "model.rotary_emb": ratio_rotary_emb,
        }

        for layer, count in special_layers.items():
            if count >= 0:
                while layers_per_device[current_device] == 0:
                    current_device += 1
                device_map[layer] = devices_idx[current_device]
                layers_per_device[current_device] -= count

        # 分配普通层
        for layer in layers:
            while layers_per_device[current_device] == 0:
                current_device += 1
            device_map[layer] = devices_idx[current_device]
            layers_per_device[current_device] -= 1

        device_map["model.embed_tokens"] = devices_idx[0]

        return device_map

    # 使用手动创建的 device_map
    device_map = create_manual_device_map(model, num_devices, is_distillation)

    # 打印 device_map 结果
    # 打印 device_map 结果和每个设备上的元素统计
    device_stats = {}
    for module, device in device_map.items():
        if device not in device_stats:
            device_stats[device] = {"count": 0, "modules": []}
        device_stats[device]["count"] += 1
        device_stats[device]["modules"].append(module)

    print("Device Map:")
    for device, stats in device_stats.items():
        print(f"Device {device}: {stats['count']} elements")
        print(f"  Modules: {', '.join(stats['modules'])}")

    del model

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype="auto",
        device_map=device_map,
        attn_implementation=(
            "eager"
            if "gemma" in model_dir.lower() or "phi" in model_dir.lower()
            else "sdpa"
        ),
        trust_remote_code=True,
    )
    return model


if __name__ == "__main__":
    ratio = [d / sum([0.8, 1, 1, 1]) for d in [0.8, 1, 1, 1]]
    balanced_load(
        "/mnt/rangehow/models/Meta-Llama-3.1-8B-Instruct",
        4,
        is_distillation=False,
        ratio=ratio,
        devices_idx=[0, 1, 2, 3],
    )

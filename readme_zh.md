本仓库包含 [NDP: Next Distribution Prediction as a More Broad Target](https://arxiv.org/abs/2408.17377) 官方实现

# 用例
## 在Qwen2.5-1.5B上进行翻译训练
我们这里以[haoranxu/ALMA-Human-Parallel](https://huggingface.co/datasets/haoranxu/ALMA-Human-Parallel)数据集为例，训练zh-en方向。

### 基于NTP的训练方式
```bash
python naive_train.py --grad
ient_accumulation_steps 64 --total_bsz 128  --dataset alma_zhen --learning_rate 5e-5 --template qwen2.5  --lr_scheduler_type cosine --warmup_ratio 0.01 --model Qwen/Qwen2.5-1.5B --w_template True
```
对于原始的NTP训练方式，我们使用huggingface的默认trainer，所以大部分参数都与hf的`TrainingArguments`保持一致。如果你需要使用自己的模型，model可以设置为模型的本地路径或者hf identifier。需要注意的是，我们项目所支持的数据集，可以在`eval/load_func.py`内查看（其中就包含我们此次需要使用的[haoranxu/ALMA-Human-Parallel](https://huggingface.co/datasets/haoranxu/ALMA-Human-Parallel)，我们将在后续部分介绍如何为我们的仓库非常便捷的适配数据集。

### 基于NDP的训练方式 
我们需要先处理数据集，使其变为NDP增强过的分布数据。
注意下面的命令需要在仓库的上层路径执行
```bash
python -m sft.ndp.preprocess_trie --model Qwen/Qwen2.5-1.5B --dataset alma_zhen
```
# 对仓库结构的简要介绍
我们的仓库支持基于NTP/知识蒸馏/NDP的训练方式。
其中基于NTP和知识蒸馏的训练方式
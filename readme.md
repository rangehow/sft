本仓库使用指南

> 未在仓库内提供，需要自行处理/下载的部分：

- sft/config.py，这个文件主要用于保存模型名/数据集名到绝对路径的映射。

```python
model_dir={
    'gemma_2b':'/data/ruanjh/best_training_method/gemma-2b'
    'llama3_8b':'/某个路径/Meta-Llama-3-8B'
}
dataset_dir={
    'alpaca_gpt4':'某个路径/alpaca-gpt4/data'
}
```

- 模型和数据集

  - 采用huggingface-cli下载在本地，切镜像，更新一下下载工具

  - ```bash
    export HF_ENDPOINT=https://hf-mirror.com # 先换国内镜像
    pip install -U huggingface_hub
    ```

  - 下载llama3-8B

  - ```bash
    huggingface-cli download --resume-download meta-llama/Meta-Llama-3-8B  --local-dir Meta-Llama-3-8B  --local-dir-use-symlinks False --exclude "original/*"
    ```

  - 下载alpaca_gpt4  

  - ```bash
    huggingface-cli download --repo-type dataset --resume-download vicgalle/alpaca-gpt4 --local-dir alpaca-gpt4 --local-dir-use-symlinks False
    ```

> 在本仓库中提供的部分

1. 数据生成阶段

   ```bash
   python preprocess.py --model llama3_8b --dataset alpaca-gpt4
   ```

2. 训练

   先设定一个accelerate config，最好使用4张卡以上，**卡数一定要是偶数**。下面可以修改的地方有num_processes，实际几张卡就写几.zero_stage,如果卡数足够多，可以尝试降低zero_stage（from 3->2->1），可以加快训练。**mixed_precision**不建议使用，deepspeed的混合精度会把所有输入数据都转成特定的dtype，导致报错，这个行为还需要再核验。

   ```yaml
   compute_environment: LOCAL_MACHINE
   debug: false
   deepspeed_config:
     gradient_accumulation_steps: 8
     offload_optimizer_device: none
     offload_param_device: none
     zero3_init_flag: false
     zero3_save_16bit_model: false
     zero_stage: 3
   distributed_type: DEEPSPEED
   downcast_bf16: 'no'
   enable_cpu_affinity: false
   machine_rank: 0
   main_training_function: main
   mixed_precision: 'no'
   num_machines: 1
   num_processes: 4
   rdzv_backend: static
   same_network: true
   tpu_env: []
   tpu_use_cluster: false
   tpu_use_sudo: false
   use_cpu: false
   ```

   然后执行训练命令，为了挂起后台，可以用tmux。

   ```bash
   CUDA_VISIBLE_DEVICES=可见的GPU_id accelerate launch --config_file my.yaml  special_train.py --model llama3_8b --dataset alpaca_gpt4  --weighted --zero_prob 0 &
   ```

   如果显卡空置充足，可以再跑下面两组：

   ```bash
   CUDA_VISIBLE_DEVICES=可见的GPU_id accelerate launch --config_file my.yaml  special_train.py --model llama3_8b --dataset alpaca_gpt4 --zero_prob 0 &
   ```

   ```bash
   CUDA_VISIBLE_DEVICES=可见的GPU_id accelerate launch --config_file my.yaml  special_train.py --model llama3_8b --dataset alpaca_gpt4 ---weighted zero_prob 0.1 &
   ```

## eval
### 常规的测试，我们采用lm-eval

安装lm-eval最新版本
```bash
pip install git+https://github.com/bigcode-project/bigcode-evaluation-harness.git
```

lm_eval 里的task list
gsm8k_cot mmlu  truthfulqa_mc2	bbh_cot_fewshot	arc_challenge drop	TriviaQA  agieval														

启动的命令大概像下面这样
```bash
CUDA_VISIBLE_DEVICES=2 lm_eval --model hf   --tasks bbh_cot_fewshot     --device cuda:0  --batch_size auto --model_args pretrained=/niutrans/NEUNLP/rjh/models/gemma-2b
```


### 代码类的测试
依赖于 bigcode-evaluation-harness,
```bash
https://github.com/bigcode-project/bigcode-evaluation-harness.git
```


MT-bench
https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md#mt-bench
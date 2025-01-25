~~~markdown
This is the official code repository for [NDP: Next Distribution Prediction as a More Broad Target](https://arxiv.org/abs/2408.17377).

# Brief Introduction to the Repository Structure

> Components not provided in the repository (need to be created/downloaded manually):

- **sft/config.py**: This file is used to map model and dataset names to their absolute paths.

```python
model_dir = {
    'gemma_2b': '/path_to_gemma-2b',
    'llama3_8b': '/some/path/Meta-Llama-3-8B'
}
dataset_dir = {
    'alpaca_gpt4': '/some/path/alpaca-gpt4/data'
}
~~~

- **Models and Datasets**

  - Download the models and datasets locally using `huggingface-cli`. Update to the latest tool and switch to a mirror if needed.

  ```bash
  export HF_ENDPOINT=https://hf-mirror.com  # Switch to a local mirror
  pip install -U huggingface_hub
  ```

  - **Download llama3-8B**:

  ```bash
  huggingface-cli download --resume-download meta-llama/Meta-Llama-3-8B \
    --local-dir Meta-Llama-3-8B \
    --local-dir-use-symlinks False --exclude "original/*"
  ```

  - **Download alpaca_gpt4**:

  ```bash
  huggingface-cli download --repo-type dataset \
    --resume-download vicgalle/alpaca-gpt4 \
    --local-dir alpaca-gpt4 \
    --local-dir-use-symlinks False
  ```

------

> Components provided in the repository:

1. **Data Preprocessing**

   ```bash
   python preprocess.py --model llama3_8b --dataset alpaca-gpt4
   ```

2. **Training**

   First, set up an `accelerate` configuration. It is recommended to use at least 4 GPUs, and the number of GPUs **must be even**. The configurable fields include `num_processes` (number of GPUs) and `zero_stage`. If you have more GPUs, you can try lowering the `zero_stage` (from 3 → 2 → 1) to speed up training. **Avoid using mixed precision**, as DeepSpeed's mixed precision implementation can cause errors when converting all input data to a specific dtype. This issue needs further verification.

   Example `accelerate` configuration (`my.yaml`):

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

   Run the training command using `tmux` to keep it running in the background:

   ```bash
   CUDA_VISIBLE_DEVICES=<GPU_IDS> accelerate launch --config_file my.yaml \
     special_train.py --model llama3_8b --dataset alpaca_gpt4 --weighted --zero_prob 0 &
   ```

   If you have additional GPUs, you can run the following additional commands:

   ```bash
   CUDA_VISIBLE_DEVICES=<GPU_IDS> accelerate launch --config_file my.yaml \
     special_train.py --model llama3_8b --dataset alpaca_gpt4 --zero_prob 0 &
   ```

   ```bash
   CUDA_VISIBLE_DEVICES=<GPU_IDS> accelerate launch --config_file my.yaml \
     special_train.py --model llama3_8b --dataset alpaca_gpt4 --weighted --zero_prob 0.1 &
   ```

------

## Evaluation

### General Testing with `lm-eval`

Install the latest version of `lm-eval`:

```bash
pip install git+https://github.com/bigcode-project/bigcode-evaluation-harness.git
```

Task list in `lm-eval` includes:

- `gsm8k_cot`
- `mmlu`
- `truthfulqa_mc2`
- `bbh_cot_fewshot`
- `arc_challenge`
- `drop`
- `TriviaQA`
- `agieval`

Example command to launch evaluation:

```bash
CUDA_VISIBLE_DEVICES=2 lm_eval --model hf --tasks bbh_cot_fewshot \
  --device cuda:0 --batch_size auto --model_args pretrained=/path/to/gemma-2b
```

------

### Code Testing

For testing code, rely on the `bigcode-evaluation-harness`, mainly focusing on the `humaneval` benchmark.

Clone the repository and following the instruction in it:

```bash
https://github.com/bigcode-project/bigcode-evaluation-harness.git
```




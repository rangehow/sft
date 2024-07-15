---
dataset_info:
  features:
  - name: instruction
    dtype: string
  - name: input
    dtype: string
  - name: output
    dtype: string
  - name: text
    dtype: string
  splits:
  - name: train
    num_bytes: 88566301
    num_examples: 52002
  download_size: 48393562
  dataset_size: 88566301
task_categories:
- text-generation
- conversational
- question-answering
language:
- en
size_categories:
- 10K<n<100K
license: cc-by-nc-4.0
tags:
- gpt4
- alpaca
- instruction-finetuning
- synthetic
---
# Dataset Card for "alpaca-gpt4"

This dataset contains English Instruction-Following generated by GPT-4 using Alpaca prompts for fine-tuning LLMs.

The dataset was originaly shared in this repository: https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM. This is just a wraper for compatibility with huggingface's datasets library.

## Dataset Description

- **Homepage:** https://instruction-tuning-with-gpt-4.github.io
- **Repository:** https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM
- **Paper:** https://arxiv.org/abs/2304.03277

## Dataset structure

It contains 52K instruction-following data generated by GPT-4 using the same prompts as in Alpaca.
The dataset has the same format as Alpaca data, except the output is generated by GPT-4:

    - `instruction`: `str`, describes the task the model should perform. Each of the 52K instructions is unique.
    - `input`: `str`, optional context or input for the task. 
    - `output`: `str`, the answer to the instruction as generated by `GPT-4`.
    - `text`: `str`, all the previous fields concatenated together, plus the same prompt used in Alpaca at the beginnig.

## Difference with the original Alpaca dataset

The original Alpaca dataset used text-davinci-003 to complete the prompts. This dataset uses those same prompts, but generating the completions with GPT-4. Thus, in general, the responses are of higher quality and lenght. Here is an example:


#### Example from Alpaca-GPT4:

```bash
{'instruction': 'Identify the odd one out.',
 'input': 'Twitter, Instagram, Telegram',
 'output': 'The odd one out is Telegram. Twitter and Instagram are social media platforms mainly for sharing information, images and videos while Telegram is a cloud-based instant messaging and voice-over-IP service.',
 'text': 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nIdentify the odd one out.\n\n### Input:\nTwitter, Instagram, Telegram\n\n### Response:\nThe odd one out is Telegram. Twitter and Instagram are social media platforms mainly for sharing information, images and videos while Telegram is a cloud-based instant messaging and voice-over-IP service.'}
```

#### Same example from original Alpaca:

```bash
{'instruction': 'Identify the odd one out.',
 'input': 'Twitter, Instagram, Telegram',
 'output': 'Telegram',
 'text': 'Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\nIdentify the odd one out.\n\n### Input:\nTwitter, Instagram, Telegram\n\n### Response:\nTelegram'}
```

## Licensing Information

The dataset is available under the [Creative Commons NonCommercial (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/legalcode).
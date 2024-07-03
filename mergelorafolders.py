import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from argparse import ArgumentParser
import os
from peft import PeftModelForCausalLM
import torch


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--path",
        required=True,
        help="Path to the directory containing checkpoint-* folders",
    )
    return parser.parse_args()


def process_checkpoint(checkpoint_path):
    with open(os.path.join(checkpoint_path, "adapter_config.json")) as o:
        base_model = json.load(o)["base_model_name_or_path"]

    print(base_model, checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(checkpoint_path)
    base_model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto")
    model = PeftModelForCausalLM.from_pretrained(base_model, checkpoint_path)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(checkpoint_path)


def main():
    args = parse_args()
    base_path = args.path

    for folder_name in os.listdir(base_path):
        if folder_name.startswith("checkpoint-"):
            checkpoint_path = os.path.join(base_path, folder_name)
            if os.path.isdir(checkpoint_path):
                process_checkpoint(checkpoint_path)


if __name__ == "__main__":
    main()

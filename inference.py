# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import gemma_dir
import torch


model_dir=gemma_dir
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)

input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt")
import pdb
pdb.set_trace()
outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))

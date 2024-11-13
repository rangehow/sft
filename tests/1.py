from transformers import AutoTokenizer


tokenizer= AutoTokenizer.from_pretrained("/mnt/rangehow/models/Qwen2.5-7B-Instruct")
input=[525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 27473, 8453, 1119, 6364, 510, 17, 23, 92015, 114396, 99250, 99879, 99561, 34204, 100052, 110734, 101949, 105428, 151645, 198, 151644, 77091, 198]
print(tokenizer.convert_ids_to_tokens(input))
print(tokenizer.decode(input))
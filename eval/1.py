
import json
shot_data_path = "sft/eval/mmlu_few_shot_promot.json"
with open(shot_data_path, "r") as file:
    cot_prompts = json.load(file)


merged_dict = {}
for d in cot_prompts:
    merged_dict.update(d)

print(merged_dict)
with open(shot_data_path, "w") as file:
    json.dump(merged_dict, file, ensure_ascii=False, indent=2)

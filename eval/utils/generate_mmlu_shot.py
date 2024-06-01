import json
import re
import datasets
from loguru import logger
from tqdm import tqdm
choices = ["A", "B", "C", "D"]


@logger.catch
def main():
    # d = datasets.load_dataset("cais/mmlu")
    json_data = {}
    domain = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ]
    for d in tqdm(domain):

        prompt = (
            f"The following are multiple choice questions (with answers) about {d}.\n\n"
        )
        domain_dataset = datasets.load_dataset("cais/mmlu", d)["dev"]
        for i in range(5):
            prompt += f"Question: {domain_dataset[i]['question']}\n"

            for j in range(len(domain_dataset[i]["choices"])):
                prompt += f"{choices[j]}. {domain_dataset[i]['choices'][j]}\n"
            prompt += f"Answer: {choices[domain_dataset[i]['answer']]}\n\n"
        json_data[d]=prompt

    with open("mmlu_few_shot_promot.json", "w", encoding="utf-8") as o:
        json.dump(json_data, o, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

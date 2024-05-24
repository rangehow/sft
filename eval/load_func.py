import datasets
dname2load = {}


def register2dict(name):
    def decorator(func):
        dname2load[name] = func
        return func

    return decorator



@register2dict(name="gsm8k")
def gsm8k():
    return datasets.load_dataset('gsm8k', "main")["test"]
@register2dict(name="mmlu")
def mmlu():
    _SUBJECTS = ["abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge", "college_biology", "college_chemistry", "college_computer_science", "college_mathematics", "college_medicine", "college_physics", "computer_security", "conceptual_physics", "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic", "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science", "high_school_european_history", "high_school_geography", "high_school_government_and_politics", "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics", "high_school_physics", "high_school_psychology", "high_school_statistics", "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality", "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management", "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition", "philosophy", "prehistory", "professional_accounting", "professional_law", "professional_medicine", "professional_psychology", "public_relations", "security_studies", "sociology", "us_foreign_policy", "virology", "world_religions"]
    #_SUBJECTS = ["abstract_algebra"]
    datasets_list = []

    for sub in _SUBJECTS:
        data = datasets.load_dataset('mmlu', sub)['test']
        # 为input列表的每个元素前面添加问题类型
        new_data = data.map(lambda x: {"input": f"{sub} {x['input']}"}, batched=False)

        datasets_list.append(new_data)

    combined_dataset = datasets.concatenate_datasets(datasets_list)

    return combined_dataset

import os
import re
import torch
from config import *
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOTrainer, GRPOConfig

SYSTEM_PROMPT = """
Yanıtlarını şu formatta ver:
<sebep>
...
</sebep>
<cevap>
...
</cevap>
"""

XML_COT_FORMAT = """\
<sebep>
{reasoning}
</sebep>
<cevap>
{answer}
</cevap>
"""

def extract_final_answer(text) -> float:
    """Extracts the final answer from the completion text."""
    numbers_found = re.findall(r'\b\d+(?:[.,]\d+)?\b', normalize_numbers(text))
    return float(numbers_found[-1]) if numbers_found else None

def normalize_numbers(text):
    """replace comma-separated numbers with dot-separated decimal numbers"""
    return re.sub(r'(\d+),(\d+)', r'\1.\2', text)

def get_gsm8ktr_questions() -> Dataset:
    """Returns the questions from the GSM8K-TR dataset."""
    data = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    # Filter out examples without numbers in the answer
    data = data.filter(lambda example: bool(re.search(r'\d+', example["answer"])))
    # Format chat template
    data = data.map(lambda row: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': row['question']}
        ],
        'answer': extract_final_answer(row['answer'])
    }, remove_columns=["question"])
    return data

dataset = get_gsm8ktr_questions()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_final_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<sebep>\n.*?\n</sebep>\n<cevap>\n.*?\n</cevap>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<sebep>.*?</sebep>\s*<cevap>.*?</cevap>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<sebep>\n") == 1:
        count += 0.125
    if text.count("\n</sebep>\n") == 1:
        count += 0.125
    if text.count("\n<cevap>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</cevap>\n")[-1])*0.001
    if text.count("\n</cevap>") == 1:
        count += 0.125
        count -= (len(text.split("\n</cevap>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

base_model = AutoModelForCausalLM.from_pretrained('./sft_checkpoint')
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

lora_config = LoraConfig(**LORA_CONFIG)
lora_model = get_peft_model(base_model, lora_config)
lora_model.to(device)

training_args = GRPOConfig(
    output_dir=GRPO_CHECKPOINT_DIR,
    **GRPO_TRAINING_CONFIG,
)

trainer = GRPOTrainer(
    model=lora_model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=[
        correctness_reward_func,
        strict_format_reward_func,
        soft_format_reward_func,
        xmlcount_reward_func,
    ],
)
trainer.model.to(device)

if torch.backends.mps.is_available():
    torch.mps.empty_cache()

checkpoint_exists = os.path.exists(GRPO_CHECKPOINT_DIR) and any(
    "checkpoint" in file for file in os.listdir(GRPO_CHECKPOINT_DIR)
)

trainer.train(resume_from_checkpoint=checkpoint_exists)
trainer.save_model(GRPO_OUTPUT_DIR)

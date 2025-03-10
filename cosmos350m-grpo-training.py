import os
import re
import torch
from scipy.spatial.distance import cosine
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
from trl import GRPOTrainer, GRPOConfig
import wandb
from config import *

wandb.login()

dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)

def preprocess_function(examples):
    return {
        "prompt": examples["question"],
        "completion": examples["answer"]
    }

dataset = dataset.map(preprocess_function, remove_columns=["question"])

def contains_numbers(example):
    """Returns True if the answer contains at least one numeric value."""
    return bool(re.search(r'\d+', example["answer"]))

dataset = dataset.filter(contains_numbers)

sentence_encoder = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)

def reasoning_similarity(response, reference):
    embedding1 = sentence_encoder.encode(response)
    embedding2 = sentence_encoder.encode(reference)
    return 1 - cosine(embedding1, embedding2)

def extract_numbers(text):
    return re.findall(r'\b\d+(?:[.,]\d+)?\b', text)

def normalize_numbers(text):
    return re.sub(r'(\d+),(\d+)', r'\1.\2', text)

def reward_hybrid(completions, **kwargs):
    """Hybrid reward function for GSM8K-TR using final answer and reasoning similarity."""
    rewards = []
    answers = kwargs.get("answer", [""] * len(completions))

    for completion, answer in zip(completions, answers):
        response_nums = extract_numbers(normalize_numbers(completion))
        answer_nums = extract_numbers(normalize_numbers(answer))
        
        if response_nums == answer_nums:
            answer_reward = 1.0
        elif set(response_nums) & set(answer_nums):
            num_match_score = len(set(response_nums) & set(answer_nums)) / max(len(response_nums), len(answer_nums))
            answer_reward = 0.8 * (num_match_score ** 1.5)
        else:
            answer_reward = 0.0
        
        final_reward = (
            REWARD_WEIGHTS["answer"] * answer_reward
        ) + (
            REWARD_WEIGHTS["reasoning_similarity"] * reasoning_similarity(completion, answer)
        )
        rewards.append(final_reward)

    return rewards

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)

lora_config = LoraConfig(**LORA_CONFIG)
lora_model = get_peft_model(base_model, lora_config)
lora_model.to(device)

training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    **TRAINING_CONFIG,
)

trainer = GRPOTrainer(
    model=lora_model,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=reward_hybrid,
)
trainer.model.to(device)

if torch.backends.mps.is_available():
    torch.mps.empty_cache()

checkpoint_exists = os.path.exists(OUTPUT_DIR) and any(
    "checkpoint" in file for file in os.listdir(OUTPUT_DIR)
)

trainer.train(resume_from_checkpoint=checkpoint_exists)

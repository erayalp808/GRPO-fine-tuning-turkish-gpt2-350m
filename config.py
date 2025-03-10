import os
import torch
from transformers import logging

# Environment Variables
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
logging.set_verbosity_info()

# Model and Dataset Configurations
BASE_MODEL_NAME = "ytu-ce-cosmos/turkish-gpt2-medium-350m-instruct-v0.1"
DATASET_NAME = "ytu-ce-cosmos/gsm8k_tr"
DATASET_SPLIT = "train"
OUTPUT_DIR = "cosmosGPT2_grpo_math_enhanced_model"

# Training Arguments
TRAINING_CONFIG = {
    "max_steps": 1000,
    "num_generations": 6,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-6,
    "bf16": True,
    "logging_steps": 20,
    "save_strategy": "steps",
    "save_steps": 50,
    "report_to": ["wandb"],
    "max_completion_length": 256,
    "max_prompt_length": 128,
    "max_grad_norm": 1.0,
}

REWARD_WEIGHTS = {
    "answer": 0.6,
    "reasoning_similarity": 0.4,
}

# LoRA Configuration
LORA_CONFIG = {
    "r": 32,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# Device Configuration
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Sentence Transformer Model for Reasoning Similarity
SENTENCE_TRANSFORMER_MODEL = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"

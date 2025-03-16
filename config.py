import os
import torch
from transformers import logging
import wandb

wandb.login()

# Environment Variables
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
logging.set_verbosity_info()

# Model and Dataset Configurations
BASE_MODEL_NAME = "ytu-ce-cosmos/turkish-gpt2-medium-350m-instruct-v0.1"
DATASET_NAME = "ytu-ce-cosmos/gsm8k_tr"
DATASET_SPLIT = "train"
GRPO_CHECKPOINT_DIR = "checkpoint_cosmosGPT2_grpo_math_enhanced_model"
GRPO_OUTPUT_DIR = "output_cosmosGPT2_grpo_math_enhanced_model"
SFT_CHECKPOINT_DIR = "checkpoint_cosmosGPT2_sft_gsmk8k_tr"
SFT_OUTPUT_DIR = "output_cosmosGPT2_sft_gsmk8k_tr"


# Training Arguments
GRPO_TRAINING_CONFIG = {
    "num_train_epochs": 1,
    "num_generations": 6,
    "per_device_train_batch_size": 6,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-6,
    "lr_scheduler_type": "cosine",
    "bf16": True,
    "logging_steps": 1,
    "save_strategy": "steps",
    "save_steps": 20,
    "report_to": ["wandb"],
    "max_completion_length": 256,
    "max_prompt_length": 256,
    "max_grad_norm": 1.0,
}

SFT_TRAINING_CONFIG = {
    "learning_rate": 5e-6,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 2,
    "bf16": True,
    "adam_beta1": 0.9,
    "adam_beta2": 0.99,
    "weight_decay": 0.1,
    "warmup_ratio": 0.1,
    "num_train_epochs": 2,
    "lr_scheduler_type": "cosine",
    "save_strategy": "epoch",
    "logging_steps": 1,
    "evaluation_strategy": "no",
    "eval_strategy": "steps",
    "report_to": ["wandb"],
    "eval_steps": 5,
    "max_grad_norm": 1.0,
}

# LoRA Configuration
LORA_CONFIG = {
    "r": 32,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# Device Configuration
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

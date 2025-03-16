# GSM8K-TR Model Training

This project fine-tunes a Turkish GPT-2 Medium model on the GSM8K-TR dataset using Supervised Fine-Tuning (SFT) and further refines it with a reward-based fine-tuning approach using Group Relative Policy Optimization (GRPO). 

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training](#training)
  - [Supervised Fine-Tuning (SFT)](#supervised-fine-tuning-sft)
  - [Group Relative Policy Optimization (GRPO)](#group-relative-policy-optimization-grpo)
- [Reward Functions](#reward-functions)
- [Model Saving](#model-saving)
- [License](#license)

## Overview
This project aims to enhance the mathematical reasoning ability of a Turkish GPT-2 Medium model using a two-stage training approach:
1. **Supervised Fine-Tuning (SFT)**: The model is fine-tuned on the GSM8K-TR dataset to learn question-answering patterns.
2. **Group Relative Policy Optimization (GRPO)**: A reward model is used to further improve reasoning and structured response generation.

## Features
- Fine-tunes a Turkish GPT-2 Medium model.
- Implements structured response formats using XML-style tags.
- Uses reward-based training with multiple reward functions.
- Supports LoRA-based parameter-efficient fine-tuning.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/gsm8k-tr-training.git
   cd gsm8k-tr-training
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have the required dataset available.

## Environment Setup

Before running the training scripts, set up your environment properly:

Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```
Install required packages:
```bash
pip install -r requirements.txt
```

## Usage
To train the model, configure `config.py` and run the corresponding training scripts.

## Configuration
Modify `config.py` to set paths, model names, and training hyperparameters. Key configuration values:
- `DATASET_NAME`: The dataset identifier.
- `BASE_MODEL_NAME`: Pretrained GPT-2 Medium model.
- `LORA_CONFIG`: LoRA parameters.
- `GRPO_TRAINING_CONFIG`: GRPO-specific training settings.
- `SFT_TRAINING_CONFIG`: SFT-specific training settings.
- `GRPO_CHECKPOINT_DIR`: Path for saving GRPO checkpoints.
- `SFT_CHECKPOINT_DIR`: Path for saving SFT checkpoints.

## Training
### Supervised Fine-Tuning (SFT)
```bash
python sft_training.py
```
This will fine-tune the base model with question-answer pairs.

### Group Relative Policy Optimization (GRPO)
```bash
python grpo_training.py
```
This will refine the SFT model using reward-based learning.

## Reward Functions
The GRPO training script includes four reward functions:
1. **Correctness Reward (`correctness_reward_func`)**: Rewards correct answers.
2. **Strict Format Reward (`strict_format_reward_func`)**: Ensures responses follow the XML template.
3. **Soft Format Reward (`soft_format_reward_func`)**: Looser format validation.
4. **XML Count Reward (`xmlcount_reward_func`)**: Encourages proper XML structure.

## Model Saving
Trained models are saved to:
- `SFT_OUTPUT_DIR` for supervised fine-tuning
- `GRPO_OUTPUT_DIR` for reinforcement fine-tuning


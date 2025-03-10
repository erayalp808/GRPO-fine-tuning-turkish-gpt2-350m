# CosmosGPT2-GRPO: Enhancing Mathematical Reasoning in Turkish GPT-2

## Overview
This project fine-tunes the [ytu-ce-cosmos/turkish-gpt2-medium-350m-instruct-v0.1](https://huggingface.co/ytu-ce-cosmos/turkish-gpt2-medium-350m-instruct-v0.1) model using [GSM8K-TR](https://huggingface.co/datasets/ytu-ce-cosmos/gsm8k_tr) with the GRPO (Group Relative Policy Optimization) framework. The goal is to improve the model's mathematical reasoning capabilities by incorporating a hybrid reward function based on numerical correctness and reasoning similarity.

## Features
- **Fine-tuning with GRPO**: Uses reward-based optimization for better reasoning.
- **Hybrid Reward Function**: Evaluates responses based on numerical accuracy and reasoning similarity.
- **LoRA (Low-Rank Adaptation)**: Efficient parameter-efficient fine-tuning method.
- **Sentence Embedding for Similarity**: Uses `sentence-transformers/all-MiniLM-L6-v2` for reasoning-based reward calculation.
- **Multi-device Training Support**: Compatible with CUDA, MPS, and CPU.
- **Integrated with W&B**: Tracks training progress using Weights & Biases.

## Installation
### Requirements
Ensure you have Python 3.8+ and install dependencies:
```sh
pip install torch transformers datasets sentence-transformers peft trl wandb scipy
```

## Environment Setup
To set up a dedicated environment for this project, follow these steps:

### Using `venv` (Recommended for Local Development)
```sh
python -m venv cosmosgpt2-env
source cosmosgpt2-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Ensure that your environment is activated before running any scripts.

## Configuration
Modify `config.py` to adjust:
- **Model and dataset**:
  - `BASE_MODEL_NAME`: Pretrained model name.
  - `DATASET_NAME`: Fine-tuning dataset.
- **Training Parameters**:
  - `TRAINING_CONFIG`: Adjust training steps, batch size, learning rate, etc.
- **LoRA Settings**:
  - `LORA_CONFIG`: Modify rank, dropout, and bias settings.

## Usage
### 1. Set Up Weights & Biases
Log into W&B before training:
```sh
wandb login
```

### 2. Train the Model
Run the main script to start training:
```sh
python main.py
```
If training was interrupted, it will resume from the last checkpoint.

## Reward Function Details
The reward function evaluates completions based on:
1. **Numerical Accuracy**: Extracts numbers and compares them with the ground truth.
2. **Reasoning Similarity**: Computes the cosine similarity of sentence embeddings.
3. **Final Reward Calculation**:
   ```
   final_reward = (0.6 * answer_reward) + (0.4 * reasoning_reward)
   ```
   where `answer_reward` is based on numerical correctness, and `reasoning_reward` is a reasoning format similarity score.

## Model Training Workflow
1. Load dataset and preprocess.
2. Filter questions containing numerical values.
3. Fine-tune the GPT-2 model using LoRA and GRPO.
4. Use the hybrid reward function for optimization.
5. Save model checkpoints at specified steps.

## Performance Tracking
- **Logging with W&B**: Monitors training metrics and logs progress.
- **Checkpointing**: Saves model every 25 steps to `OUTPUT_DIR`.

## Device Support
The model runs on the best available hardware:
- **CUDA (NVIDIA GPU)**
- **MPS (Mac M-series GPU)**
- **CPU (Fallback)**


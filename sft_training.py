import re
from config import *
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
dataset = dataset.filter(lambda example: bool(re.search(r'\d+', example["answer"])))
dataset = dataset.map(lambda row: {
    'messages': [
        {'role': 'user', 'content': row['question']},
        {'role': 'assistant', 'content': row['answer']}
    ]
}, remove_columns=['question', 'answer'])
split_dataset = dataset.select(range(1250)).train_test_split(test_size=0.2)

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

training_args = SFTConfig(
    output_dir=SFT_CHECKPOINT_DIR,
    **SFT_TRAINING_CONFIG
)

trainer = SFTTrainer(
    model=base_model,
    args=training_args,
    train_dataset=split_dataset["train"],
    eval_dataset=split_dataset["test"],
    processing_class=tokenizer
)
trainer.model.to(device)

trainer.train()

trainer.save_model(SFT_OUTPUT_DIR)
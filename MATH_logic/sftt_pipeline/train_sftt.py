import os 
import sys
import torch as th
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

run_name = 'SFT-Run_1'

import wandb
wandb.init(
    project='Into LLM Reasoning',
    name=f'GSM8K-SFTT-Test : {run_name}'
)

from models.model import get_model, get_tokenizer
from MATH_logic.dataset_utils import convert_solution_to_tagged_completion

from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig

def process_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple): 
        logits = logits[0]
    pred_ids = th.argmax(logits, dim = -1) # Shape : [B, seq_len]
    return pred_ids

def compute_metrics(eval_pred):
    predictions, labels = eval_pred.predictions, eval_pred.label_ids

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = predictions.reshape(-1)
    labels = labels.reshape(-1)

    valid = labels != -100
    predictions = predictions[valid]
    labels = labels[valid]

    accuracy = (predictions == labels).mean()
    return {'Token Accuracy': float(accuracy)}

def format_prompt(example, tokenizer):
    prompt = (
        "Solve this math problem step by step. Enclose your entire reasoning process within <thinking:step>...</thinking:step> tags.\n"
        "At the end, put the final answer inside <answer> \\boxed{} </answer>.\n"
        f"Problem: {example['problem']} \n"
    )
    completion = convert_solution_to_tagged_completion(example["solution"]) + tokenizer.eos_token 

    return {
        "prompt": prompt,
        "completion": completion
    }


# Load model and LoRA configuration
model = get_model()
tokenizer = get_tokenizer()

# Load datasets
print('Loading dataset')
datasets = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default")
train_val = datasets["train"].train_test_split(test_size=0.15, seed = 42) # Split dataset

dataset_train = train_val["train"].map(lambda example : format_prompt(example, tokenizer), remove_columns=train_val["train"].column_names)
dataset_val = train_val["test"].map(lambda example : format_prompt(example, tokenizer), remove_columns=train_val["test"].column_names)
print(dataset_train[0])
print('Train dataset has loaded')

# Configuration of SFTT
stft_args = SFTConfig(
    output_dir = 'sftt_MATH_result',

    learning_rate = 1e-6,
    warmup_steps= 0.02, 
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    max_grad_norm=0.5,

    per_device_train_batch_size = 10,
    gradient_accumulation_steps = 10, 
    num_train_epochs=1,

    bf16=True,
    gradient_checkpointing=True,

    max_length=1024,
    completion_only_loss=True,

    report_to='wandb',
    logging_strategy='steps',
    logging_steps=2,

    eval_strategy='epoch',
    save_strategy='steps',
    save_steps=50, 

    load_best_model_at_end=False,
    deepspeed="ds_config.json"
)

# Init the SFTT trainer
trainer = SFTTrainer(
    model = model, 
    train_dataset = dataset_train,
    eval_dataset=dataset_val,
    args = stft_args,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=process_logits_for_metrics
)

trainer.train()
trainer.accelerator.wait_for_everyone()
trainer.save_model('sftt_MATH_model_v2')
wandb.finish()

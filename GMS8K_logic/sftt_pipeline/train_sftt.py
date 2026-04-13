import os 
import sys
import torch as th

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
os.environ['WANDB_API_KEY'] = ''

run_name = 'SFT-GSM8K-Run_1'

import wandb
wandb.init(
    project='Into LLM Reasoning',
    name=f'GSM8K-SFTT-Test : {run_name}'
)

from models.model import get_model

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

def format_prompt(example):
    return {"text": f"Question: {example['question']}\nAnswer: {example['answer']}"}

# Load datasets
print('Loading dataset')
dataset = load_dataset("openai/gsm8k", "main")
dataset_train = dataset['train'].map(format_prompt, remove_columns=['question', 'answer'])
dataset_eval  = dataset['test'].map(format_prompt, remove_columns=['question', 'answer'])
print('Train dataset has loaded')

# Load model and LoRA configuration
model, lora_confg = get_model()

# Configuration of SFTT
stft_args = SFTConfig(
    output_dir = 'sftt_GMS8K_result',
    learning_rate = 2e-5,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4, 
    num_train_epochs=3,
    bf16=True,
    report_to='wandb',
    gradient_checkpointing=True,
    max_length=1024,
    logging_strategy='epoch',
    eval_strategy='epoch',
    save_strategy='best',  
    load_best_model_at_end=True
)

# Init the SFTT trainer
trainer = SFTTrainer(
    model = model, 
    train_dataset = dataset_train,
    eval_dataset=dataset_eval,
    peft_config = lora_confg, 
    args = stft_args,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=process_logits_for_metrics
)

trainer.train()
wandb.finish()
trainer.model.save_pretrained('sftt_GMS8K_model')
import os 
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
os.environ['WANDB_API_KEY'] = ''

run_name = 'Run_1'

import wandb
wandb.init(
    project='Into LLM Reasoning',
    name=f'GSM8K-RLVR-Test : {run_name}'
)

import torch as th
import gc

from rewards_utils import accuracy_reward, format_reward

from model import get_model

from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from peft import PeftModel

# Load mnodel 
model, lora_confg = get_model()
model_sftt = PeftModel.from_pretrained(model, 'sftt_GMS8K_model')
model_merged = model_sftt.merge_and_unload()

del model, model_sftt
th.cuda.empty_cache()
gc.collect()

def format_prompt(example):
    return {"text": f"Question: {example['question']}\nAnswer: {example['answer']}"}

# Load datasets
print('Loading dataset')
dataset = load_dataset("openai/gsm8k", "main")
dataset_train = dataset['train'].map(format_prompt, remove_columns=['question', 'answer'])
dataset_eval  = dataset['test'].map(format_prompt, remove_columns=['question', 'answer'])
print('Train dataset has loaded')

# Configuration of GRPO 
training_args = GRPOConfig(
    output_dir='rlvr_GMS8K_result',
    learning_rate=3e-6,
    num_generations=8,
    max_completion_length = 512,
    report_to = 'wandb',
    logging_steps = 1,
    bf16=True,
    gradient_checkpointing=True,
    logging_strategy='epoch',
    eval_strategy='epoch',
    save_strategy='best',  
    load_best_model_at_end=True,
    num_train_epochs=5
)

# Init the GRPO trainer 
trainer = GRPOTrainer(
    model = model_merged,
    reward_funcs=[accuracy_reward, format_reward],
    args = training_args,
    train_dataset=dataset_train,
    peft_config=lora_confg,
)

trainer.train()
wandb.finish()
trainer.model.save_pretrained('rlvr_GMS8K_model')
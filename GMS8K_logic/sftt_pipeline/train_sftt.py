import os 
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
os.environ['WANDB_API_KEY'] = ''

run_name = 'SFT-GSM8K-Run_1'

import wandb
wandb.init(
    project='Into LLM Reasoning',
    name=f'GSM8K-SFTT-Test : {run_name}'
)

from build_dataset import map_dataset
from model import get_model

from datasets import load_dataset, Dataset
from trl import SFTTrainer, SFTConfig

# Load datasets
print('Loading dataset')
dataset = load_dataset("openai/gsm8k", "main")
dataset_train = dataset['train']
dataset_train = dataset_train.rename_column("question", "text")
print('Train dataset has loaded')

# Load model and LoRA configuration
model, lora_confg = get_model()

# Configuration of SFTT
stft_args = SFTConfig(
    output_dir = 'sftt_GMS8K_result',
    learning_rate = 2e-5,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4, 
    num_train_epochs=10,
    bf16=True,
    report_to='wandb',
    gradient_checkpointing=True,
    max_length=1024,
    logging_strategy='epoch',
    eval_strategy='epoch',
    load_best_model_at_end=True
)

# Init the SFTT trainer
trainer = SFTTrainer(
    model = model, 
    train_dataset = dataset_train,
    peft_config = lora_confg, 
    args = stft_args,
)

trainer.train()
wandb.finish()
trainer.model.save_pretrained('sftt_GMS8K_model')
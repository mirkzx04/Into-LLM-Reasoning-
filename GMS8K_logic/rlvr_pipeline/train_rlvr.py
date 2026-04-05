import os 
os.environ['WANDB_API_KEY'] = ''

run_name = ''

import wandb
wandb.init(
    project='Into LLM Reasoning',
    name=f'GSM8K-RLVR-Test : {run_name}'
)

from build_dataset import map_dataset

from rewards_utils import accuracy_reward, format_reward

from model import get_model

from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from peft import PeftModel

# Load mnodel 
model, lora_confg = get_model()
model_sftt = PeftModel.from_pretrained(model, 'sftt_GMS8K_model')

# Load datasets
dataset = load_dataset("openai/gsm8k", "main")
dataset_train = dataset['train']

# Build dataset fro training loop 
dataset_train = map_dataset(dataset_train)

# Configuration of GRPO 
training_args = GRPOConfig(
    output_dir='rlvr_GMS8K_result',
    learning_rate=3e-6,
    num_generations=8,
    map_prompt_lenght=256,
    max_completion_lenght = 512,
    report_to = 'wandb',
    logging_steps = 1,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=5
)

# Init the GRPO trainer 
trainer = GRPOTrainer(
    model = model_sftt,
    reward_funcs=[accuracy_reward, format_reward],
    args = training_args,
    train_dataset=dataset_train,
    peft_config=lora_confg,
)

trainer.train()
wandb.finish()
trainer.model.save_pretrained('rlvr_GMS8K_model')
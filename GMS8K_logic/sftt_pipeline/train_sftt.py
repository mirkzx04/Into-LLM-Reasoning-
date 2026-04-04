import os 
os.environ['WANDB_API_KEY'] = ''

run_name = ''

import wandb
wandb.init(
    project='Into LLM Reasoning',
    name=f'GSM8K-SFTT-Test : {run_name}'
)

from build_dataset import map_dataset

from model import get_model

from datasets import load_dataset
from trl import SFFTrainer, SFTTConfig

# Load datasets
dataset = load_dataset("openai/gsm8k", "main")
dataset_train = dataset['train']

# Build dataset fro training loop 
dataset_train = map_dataset(dataset_train)

# Load model and LoRA configuration
model, lora_confg = get_model()

# Configuration of SFTT
stft_args = SFTTConfig(
    output_dir = 'sftt_GMS8K_result',
    learning_rate = 2e-5,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4, 
    max_seq_length=1024,
    dataset_text_field = 'prompt',
    num_train_epochs=10,
    bf16=True,
    report_to='wandb',
    logging_steps=10,
    gradient_checkpointing=True,
)

# Init the SFTT trainer
trainer = SFFTrainer(
    model = model, 
    train_dataset = dataset_train,
    peft_config = lora_confg, 
    args = stft_args,
)

trainer.train()
wandb.finish()
trainer.model.save_pretrained('sftt_GMS8K_model')
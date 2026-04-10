import os 
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
os.environ['WANDB_API_KEY'] = ''

run_name = 'Run_1'

import wandb
import torch as th
import gc
import multiprocessing as mp

from rewards_utils import accuracy_reward, format_reward

from model import get_model

from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from peft import PeftModel

def format_prompt(example):
    prompt_txt = f"Resolve this math problem with reasoning step by step. Math problem : f{example['question']}"
    return {
        "prompt" : [
            {"role" : "user", "content" : prompt_txt}
        ],
        "answer" : example["answer"],
    }

def main():
    wandb.init(
        project='Into LLM Reasoning',
        name=f'GSM8K-RLVR-Test : {run_name}'
    )

    # Load model
    model, lora_confg = get_model()
    model_sftt = PeftModel.from_pretrained(model, 'sftt_GMS8K_model')
    model_merged = model_sftt.merge_and_unload()
    print(f'Model on : {next(model_merged.parameters()).device}')
    del model, model_sftt
    th.cuda.empty_cache()
    gc.collect()

    # Load datasets
    print('Loading dataset')
    datasets = load_dataset("openai/gsm8k", "main")
    dataset_train = datasets['train'].map(format_prompt, remove_columns=datasets["train"].column_names)
    dataset_eval  = datasets['test'].map(format_prompt, remove_columns=datasets["test"].column_names)
    print('Train dataset has loaded')

    # Configuration of GRPO 
    training_args = GRPOConfig(
        output_dir='rlvr_GMS8K_result',
        learning_rate=3e-6,

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,

        num_generations=4,
        max_completion_length = 128,
        max_prompt_length = 128,

        report_to = 'wandb',
        logging_strategy='epoch',

        bf16=True,
        gradient_checkpointing=True,
        
        eval_strategy='epoch',
        save_strategy='epoch', 

        load_best_model_at_end=True,
        num_train_epochs=5
    )

    # Init the GRPO trainer 
    trainer = GRPOTrainer(
        model = model_merged,
        reward_funcs=[accuracy_reward, format_reward],
        args = training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        peft_config=lora_confg,
    )

    print("Training has started")
    trainer.train()
    wandb.finish()
    trainer.model.save_pretrained('rlvr_GMS8K_model')

if __name__ == "__main__":
    mp.freeze_support()
    main()
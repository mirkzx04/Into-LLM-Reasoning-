import os 
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
os.environ['WANDB_API_KEY'] = ''

run_name = 'Run_3-New Concise Accuracy Reward'

import wandb
import torch as th
import gc
import multiprocessing as mp

from rewards_utils import accuracy_reward, format_reward, concise_accuracy_reward

from models.model_wrapper import gsm8k_sftt_model

from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from peft import PeftModel

device = "cuda" if th.cuda.is_available() else "cpu"

def format_prompt(example):
    prompt_txt = (
        f"Solve this math problem step by step."
        f"At the end, write only the final answer in the format #### <number>\n"
        f"problem : {example['question']}"
    )
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
    model, lora_confg = gsm8k_sftt_model()
    model = model.to(device)

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

        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=12,

        num_generations=8,
        max_completion_length = 512,

        temperature=1.1, 
        top_p = 0.95,

        scale_rewards="batch",
        loss_type="dr_grpo",
        reward_weights=[1.0, 0.1, 0.2],

        report_to = 'wandb',
        logging_strategy='steps',
        logging_steps=2,

        bf16=True,
        gradient_checkpointing=True,
        
        eval_strategy='epoch',
        save_strategy='epoch', 
        load_best_model_at_end=True,
        num_train_epochs=2
    )

    # Init the GRPO trainer 
    trainer = GRPOTrainer(
        model = model,
        reward_funcs=[accuracy_reward, format_reward, concise_accuracy_reward],
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
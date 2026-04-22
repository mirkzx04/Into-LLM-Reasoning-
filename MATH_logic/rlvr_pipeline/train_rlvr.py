import os 
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

run_name = 'Run_1'

import wandb
import torch as th
import gc
import multiprocessing as mp

from rewards_utils import acc_reward, format_reward

from models.model import get_model, get_tokenizer

from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

device = "cuda" if th.cuda.is_available() else "cpu"
print(device)

def format_prompt(example):
    prompt_txt = (
        "Solve this math problem step by step.\n"
        "Write each reasoning step inside <reasoning:step>...</reasoning:step> tags.\n"
        "At the end, put the final answer inside <answer>\\boxed{}</answer>.\n"
        f"Problem: {example['problem']}\n"
    )
    return {
        "prompt": prompt_txt,              
        "solution": example["solution"],
    }

def main():
    wandb.init(
        project='Into LLM Reasoning',
        name=f'MATH-RLVR-Test : {run_name}'
    )

    # Load model
    sftt_pth = "sftt_MATH_model_v2"
    model = get_model(sftt_pth)
    tokenizer = get_tokenizer(sftt_pth)
    model = model.train().to(device)

    th.cuda.empty_cache()
    gc.collect()

    # Load datasets
    print('Loading dataset')
    datasets = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default")
    train_val = datasets["train"].train_test_split(test_size=0.15, seed = 42) # Split dataset

    dataset_train = train_val["train"].map(format_prompt, remove_columns=train_val["train"].column_names)
    dataset_val = train_val["test"].map(format_prompt, remove_columns=train_val["test"].column_names)

    # Configuration of GRPO 
    training_args = GRPOConfig(
        output_dir='rlvr_GMS8K_result',
        learning_rate=2e-6,

        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        gradient_accumulation_steps=20,

        num_generations=10,
        max_completion_length = 1024,

        scale_rewards="batch",
        loss_type="dr_grpo",
        reward_weights=[1.0, 0.05],

        report_to = 'wandb',
        logging_strategy='steps',
        logging_steps=2,

        bf16=True,
        gradient_checkpointing=True,
        use_vllm=True,
        deepspeed="ds_config.json",
        
        eval_strategy='epoch',
        save_strategy='best', 
        load_best_model_at_end=True,
        num_train_epochs=2
    )

    # Init the GRPO trainer 
    trainer = GRPOTrainer(
        model = model,
        reward_funcs=[acc_reward, format_reward],
        args = training_args, 
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
    )
 
    print("Training has started")
    trainer.train()
    trainer.accelerator.wait_for_everyone()
    trainer.save_model('rlvr_MATH_model')
    wandb.finish()

if __name__ == "__main__":
    mp.freeze_support()
    main()
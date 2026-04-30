import os 
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["WANDB_SILENT"] = "true"

run_name = 'Run_1'

import wandb
import torch as th
import gc
import multiprocessing as mp

from rewards_utils import acc_reward, format_reward, concise_accuracy_reward
from models.model import get_model, get_tokenizer
from MATH_logic.dataset_utils.dataset_splitting import build_train_val_dataset

from trl import GRPOTrainer, GRPOConfig

device = "cuda" if th.cuda.is_available() else "cpu"
SFTT_PTH = "sftt_model_math"

def main():
    # Configuration of GRPO 
    training_args = GRPOConfig(
        output_dir="GRPO_Trainings",
        learning_rate=2e-6,

        mask_truncated_completions=True,

        per_device_train_batch_size=10,
        per_device_eval_batch_size=10,
        gradient_accumulation_steps=20,

        num_generations=10,
        max_completion_length = 1500,
        beta = 0.0,
        temperature=0.7,
        top_p=0.95,

        scale_rewards="batch",
        loss_type="dapo",
        reward_weights=[1.0, 0.03],

        report_to = 'wandb',
        logging_strategy='steps',
        logging_steps=5,

        bf16=True,
        gradient_checkpointing=True,
        
        use_vllm=True,
        vllm_gpu_memory_utilization=0.2,
        vllm_max_model_length=2000,

        use_liger_kernel=True,
        deepspeed="ds_config.json",
        
        eval_strategy='steps',
        eval_steps=50,

        save_strategy='steps',
        save_steps=50, 

        load_best_model_at_end=False,
        num_train_epochs=2
    )

    model = get_model(SFTT_PTH)
    tokenizer = get_tokenizer(SFTT_PTH)

    dataset_train, dataset_val = build_train_val_dataset(tokenizer, training="rlvr")
    
    run_name = f" Run 2 [Scale Rewards : None - Temp piu bassa - Peso Formato piu basso]"
    wandb.init(
        project='Into LLM Reasoning',
        name=f'[MATH RLVR] : {run_name}'
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
    trainer.save_model("rlvr_model")
    wandb.finish()

if __name__ == "__main__":
    mp.freeze_support()
    main()
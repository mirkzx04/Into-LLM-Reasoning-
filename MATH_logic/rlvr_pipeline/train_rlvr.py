import os 
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

run_name = 'Run_1'

import wandb
import torch as th
import gc
import multiprocessing as mp

from rewards_utils import acc_reward, format_reward, concise_accuracy_reward
from models.model import get_model, get_tokenizer
from MATH_logic.dataset_utils.dataset_splitting import build_t1_set, build_t2_set, build_t3_set

from trl import GRPOTrainer, GRPOConfig

device = "cuda" if th.cuda.is_available() else "cpu"
print(device)

SFTT_PTH = "sftt_numina"
RLVR_PTH = "rlvr_model_math"

TRAIN_SPLIT = ["T1", "T2", "T3"]

def main():
    # Configuration of GRPO 
    training_args = GRPOConfig(
        learning_rate=2e-6,

        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=20,

        num_generations=4,
        max_completion_length = 2048,
        beta = 0.01,
        temperature=0.8,

        scale_rewards="batch",
        loss_type="dr_grpo",
        reward_weights=[1.0, 0.2],

        report_to = 'wandb',
        logging_strategy='steps',
        logging_steps=2,

        bf16=True,
        gradient_checkpointing=True,
        
        use_vllm=True,
        vllm_gpu_memory_utilization=0.2,
        vllm_max_model_length=4096,

        use_liger_kernel=True,
        deepspeed="ds_config.json",
        
        eval_strategy='epoch',
        save_strategy='best', 
        load_best_model_at_end=True,
        num_train_epochs=2
    )

    for split in TRAIN_SPLIT: 
        if split == "T1" : 
            model = get_model(SFTT_PTH)
            tokenizer = get_tokenizer(SFTT_PTH)

            dataset_train, dataset_val = build_t1_set(tokenizer, type_training="rlvr")
        else : 
            model = get_model(RLVR_PTH)
            tokenizer = get_tokenizer(RLVR_PTH)

            if split == "T2":
                dataset_train, dataset_val = build_t2_set(tokenizer, type_training="rlvr")
            elif split == "T3":
                dataset_train, dataset_val = build_t3_set(tokenizer, type_training="rlvr")
        
        run_name = f"[{split}]_Run1"
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
        trainer.save_model(RLVR_PTH)
        wandb.finish()

if __name__ == "__main__":
    mp.freeze_support()
    main()
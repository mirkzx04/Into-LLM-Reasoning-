import os 
import sys
import torch as th
import wandb
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)
os.environ["WANDB_SILENT"] = "true"

from models.model import get_model, get_tokenizer
from MATH_logic.dataset_utils.dataset_splitting import build_numina_train, build_t1_set, build_t2_set, build_t3_set

from trl import SFTTrainer, SFTConfig

SFTT_PTH = "sftt_model_math"
TRAIN_SPLIT = ["numina", "T1", "T2", "T3"]

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

# Configuration of SFTT
stft_args = SFTConfig(
    learning_rate = 1e-6,
    warmup_steps= 0.02, 
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    max_grad_norm=0.5,

    per_device_train_batch_size = 10,
    gradient_accumulation_steps = 10, 
    num_train_epochs=1,

    bf16=True,
    gradient_checkpointing=True,

    max_length=1024,
    completion_only_loss=True,

    report_to='wandb',
    logging_strategy='steps',
    logging_steps=2,

    eval_strategy='epoch',
    save_strategy='steps',
    save_steps=50, 

    load_best_model_at_end=False,
    deepspeed="ds_config.json"
)

for split in TRAIN_SPLIT:
    if split == "numina" :
        model = get_model()
        tokenizer = get_tokenizer()

        dataset_train, dataset_val = build_numina_train(tokenizer)
    else:
        model = get_model(SFTT_PTH)
        tokenizer = get_tokenizer(SFTT_PTH)

        if split == "T1":
            dataset_train, dataset_val = build_t1_set(tokenizer)
        elif split == "T2":
            dataset_train, dataset_val = build_t2_set(tokenizer)
        elif split == "T3":
            dataset_train, dataset_val = build_t3_set(tokenizer)
        else:
            raise ValueError(f"Unknown split: {split}")


    run_name = f"[{split}]_Run1"
    wandb.init(
        project='Into LLM Reasoning',
        name=f'[MATH SFTT] : {run_name}'
    )

    # Init the SFTT trainer
    trainer = SFTTrainer(
        model = model, 
        train_dataset = dataset_train,
        eval_dataset=dataset_val,
        args = stft_args,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=process_logits_for_metrics
    )

    trainer.train()
    trainer.accelerator.wait_for_everyone()
    trainer.save_model(SFTT_PTH)
    wandb.finish()

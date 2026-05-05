import os 
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from pathlib import Path

from analysis.extract_activation import extract_activation, get_model, get_tokenizer
from MATH_logic.dataset_utils.dataset_splitting import build_ood_eval_dataset

ACTIVATION_DATASET_PATH = "activation_dataset"

RLVR_PATH = "rlvr_model_math"
SFT_PATH = "sftt_model_math"

def get_activation_dataset():
    gen_model = get_model(RLVR_PATH)
    gen_tokenizer = get_tokenizer(RLVR_PATH)
    gen_dataset = build_ood_eval_dataset(
        tokenizer=gen_tokenizer,
        mode="rlvr"
    )

    model_desc = [(None, "base"), (SFT_PATH, "sftt"), (RLVR_PATH, "rlvr")]
    
    return extract_activation(
        max_new_tokens=2000,
        batch_size= 40,
        gen_model= gen_model,
        gen_tokenizer= gen_tokenizer,
        gen_dataset= gen_dataset,
        model_desc= model_desc,
        generator_name="qwen5_1.5B_rlvr",
        ood_dataset_name="ood_eval_dataset",
        save_path=ACTIVATION_DATASET_PATH
    )
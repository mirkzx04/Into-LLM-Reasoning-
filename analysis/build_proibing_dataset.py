import os 
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import pandas as pd
import numpy as np

from math_verify import parse, verify
from datasets import load_dataset

from analysis.extract_activation import extract_residual_out

from models.model import get_tokenizer
from models.model_wrapper import gsm8k_rlvr_model, gsm8k_sftt_model, get_model

from GMS8K_logic.rlvr_pipeline.rewards_utils import extract_answer

platinum = load_dataset("madrylab/gsm8k-platinum", "main", split="test")

tokenizer = get_tokenizer()
rlvr_model = gsm8k_rlvr_model()
sftt_model = gsm8k_sftt_model()
base_model = get_model()

rlvr_rows = []
sftt_rows = []
base_rows = []

for q, a in zip(platinum['question'], platinum['answer']):
    gt = extract_answer(a)

    rlvr_answer = extract_answer(rlvr_model.generate(q))
    sftt_answer = extract_answer(sftt_model.generate(q))
    base_answer = extract_answer(base_model.generate(q))

    correct_rlvr = (
        rlvr_answer is not None and gt is not None
        and verify(parse(rlvr_answer), parse(gt))
    )
    correct_sftt = (
        sftt_answer is not None and gt is not None
        and verify(parse(sftt_answer), parse(gt))
    )
    correct_base = (
        base_answer is not None and gt is not None
        and verify(parse(base_answer), parse(gt))
    )

    models_residuals = extract_residual_out(last_token=True, prompts=q)

    rlvr_rows.append({
        'question' : q,
        'ground_truth' : gt,
        'model_answer' : rlvr_answer,
        'correct' : correct_rlvr,
        'resid_post' : models_residuals['rlvr']['resid_post_outs'],
        'resid_attn_mlp' : models_residuals['rlvr']['resid_attn_mlp_outs'],
        'resid_attn' : models_residuals['rlvr']['resid_attn_outs']
    })
    sftt_rows.append({
        'question' : q,
        'ground_truth' : gt,
        'model_answer' : sftt_answer,
        'correct' : correct_sftt,
        'residual_post' : models_residuals['sftt']['resid_post_outs'],
        'resid_attn_mlp' : models_residuals['sftt']['resid_attn_mlp_outs'],
        'resid_attn' : models_residuals['sftt']['resid_attn_outs']
    })
    base_rows.append({
        'question' : q,
        'ground_truth' : gt,
        'model_answer' : base_answer,
        'correct' : correct_base,
        'residual_post' : models_residuals['base']['resid_post_outs'],
        'resid_attn_mlp' : models_residuals['base']['resid_attn_mlp_outs'],
        'resid_attn' : models_residuals['base']['resid_attn_outs']
    })

df_rlvr = pd.DataFrame(rlvr_rows)
df_sftt = pd.DataFrame(sftt_rows)
df_base = pd.DataFrame(base_rows)
    

    


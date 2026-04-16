import os 
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import pandas as pd
import torch as th

from math_verify import parse, verify
from datasets import load_dataset
from tqdm import tqdm

from analysis.extract_activation import (
    extract_residual_out, 
    hf_base, 
    hf_sftt,
    hf_rlvr,
    tokenizer
)

from GMS8K_logic.rlvr_pipeline.rewards_utils import extract_answer

def generate_text(model, tokenizer, prompt, max_tokens = 512):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors = 'pt').to(device)

    with th.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens = max_tokens,
            do_sample = False,
            pad_token_id = tokenizer.pad_token_id,
            eos_token_id = tokenizer.eos_token_id
        )
    new_tokens = out[0, inputs['input_ids'].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens = True)

device = 'cuda' if th.cuda.is_available() else 'cpu'

platinum = load_dataset("madrylab/gsm8k-platinum", "main", split="test")

rlvr_rows = []
sftt_rows = []
base_rows = []

for q, a in tqdm(zip(platinum['question'], platinum['answer'])):
    gt = extract_answer(a)

    rlvr_answer = extract_answer(generate_text(hf_rlvr, tokenizer, q))
    sftt_answer = extract_answer(generate_text(hf_sftt, tokenizer, q))
    base_answer = extract_answer(generate_text(hf_base, tokenizer, q))

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
        'resid_pre' : models_residuals['rlvr']['resid_pre_outs'],
        'resid_attn_mlp' : models_residuals['rlvr']['resid_attn_mlp_outs'],
        'resid_attn' : models_residuals['rlvr']['resid_attn_outs']
    })
    sftt_rows.append({
        'question' : q,
        'ground_truth' : gt,
        'model_answer' : sftt_answer,
        'correct' : correct_sftt,
        'resid_pre' : models_residuals['sftt']['resid_pre_outs'],
        'resid_attn_mlp' : models_residuals['sftt']['resid_attn_mlp_outs'],
        'resid_attn' : models_residuals['sftt']['resid_attn_outs']
    })
    base_rows.append({
        'question' : q,
        'ground_truth' : gt,
        'model_answer' : base_answer,
        'correct' : correct_base,
        'resid_pre' : models_residuals['base']['resid_pre_outs'],
        'resid_attn_mlp' : models_residuals['base']['resid_attn_mlp_outs'],
        'resid_attn' : models_residuals['base']['resid_attn_outs']
    })

pd.DataFrame(rlvr_rows).to_pickle('df_rlvr.pkl')
pd.DataFrame(sftt_rows).to_pickle('df_sftt.pkl')
pd.DataFrame(base_rows).to_pickle('df_base.pkl')
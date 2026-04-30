import os 
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch as th

import models.rope_theta

from transformers import AutoConfig
from transformer_lens import HookedTransformer
from pathlib import Path

from models.model import MODEL_ID, get_model, get_tokenizer

# seting models dir and device
device = 'cuda' if th.cuda.is_available() else 'cpu'

SFT_PTH = "sftt_model_math"
RLVR_PTH = "rlvr_model_math"

# Instance models
hf_base = get_model()
base_tokenizer = get_tokenizer()

hf_sftt = get_model(SFT_PTH)
sft_tokenizer = get_tokenizer(SFT_PTH)

hf_rlvr = get_model(RLVR_PTH)
rlvr_tokenizer = get_tokenizer(RLVR_PTH)

tokenizer = get_tokenizer()

base_tl = HookedTransformer.from_pretrained_no_processing(
    MODEL_ID,
    hf_model = hf_base,
    tokenizer = tokenizer,
    device = device,
    dtype=th.bfloat16,
)
del hf_base
th.cuda.empty_cache()

sff_tl = HookedTransformer.from_pretrained_no_processing(
    MODEL_ID,
    hf_model = hf_sftt,
    tokenizer = tokenizer,
    device = device,
    dtype=th.bfloat16,
)
del hf_sftt
th.cuda.empty_cache()

rlvr_tl = HookedTransformer.from_pretrained_no_processing(
    MODEL_ID,
    hf_model = hf_rlvr,
    tokenizer = tokenizer,
    device = device,
    dtype=th.bfloat16,
)
del hf_rlvr
th.cuda.empty_cache()

models = [
    (base_tl, 'base'), 
    (sff_tl, 'sftt'), 
    (rlvr_tl, 'rlvr')
]

def forward_with_cache(model, prompts, last_token, names_filter = None):
    with th.no_grad():
        _, cache = model.run_with_cache(
            prompts,
            names_filter = names_filter,
            pos_slice = -1 if last_token else None
        ) # Forward with caching to extract representation

    return cache

def get_hooks(interesting_layers, blocks):
    return [
        f"blocks.{l}.{b}"
        for l in interesting_layers
        for b in blocks
    ]

def extract_mlp_attn_out(prompts, interesting_layers, blocks, last_token = True):
    models_act = {}

    hook_names = get_hooks(interesting_layers, blocks)

    for model, model_name in models:
        model_cache = forward_with_cache(model, prompts, last_token, hook_names)
        layer_act = {}

        # Extract activation
        for l in interesting_layers:
            # Shape : [B, d_model] or [B, pos, d_mod]
            layer_act[l] = {
                block : model_cache[f"blocks.{l}.{block}"] 
                for block in blocks
            } 

        models_act[model_name] = layer_act

    return models_act
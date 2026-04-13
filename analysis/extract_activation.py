import os 
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch as th
from datasets import load_dataset

from transformer_lens import HookedTransformer

from models.model_wrapper import gsm8k_rlvr_model, gsm8k_sftt_model, vanilla_model

# seting models and device
device = 'cuda' if th.cuda.is_available() else 'cpu'
rlvr_model = gsm8k_rlvr_model()
sftt_model = gsm8k_sftt_model()
base_model = vanilla_model()

# Setting system prompt
SYSTEM_PROMPT = 'Solve this math problem thinking step by step'

def forward_with_cache(model, prompt):
    with th.no_grad():
        _, cache = model.run_with_cache(model, prompt) # Forward with caching to extract representation

    return cache

def extract_mlp_attn(model, prompt, last_token = True):
    n_layers = model.cfg.n_layers

    attn_outputs = {}
    mlp_outputs = {}

    model_cache = forward_with_cache(model, prompt)

    # Extract attn and mlp representation for each layers
    for l in range(n_layers):
        if last_token : 
            # Shape : [B, d_model]
            attn_out = model_cache['attn_out', l][:, -1, :].detach().cpu()
            mlp_out = model_cache['mlp_out', l][:, -1, :].detach().cpu()
        else :
            # Shape : [B, pos, d_model]
            attn_out = model_cache['attn_out', l].detach().cpu()
            mlp_out = model_cache['mlp_out', l].detach().cpu()

        attn_outputs[l] = attn_out
        mlp_outputs[l] = mlp_out

    return attn_outputs, mlp_outputs


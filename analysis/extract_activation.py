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
models = [(base_model, 'base'), (sftt_model, 'sftt'), (rlvr_model, 'rlvr')]

# Setting system prompt
SYSTEM_PROMPT = 'Solve this math problem thinking step by step'
prompts = []

def forward_with_cache(model):
    with th.no_grad():
        _, cache = model.run_with_cache(prompts) # Forward with caching to extract representation

    return cache

def extract_mlp_attn(last_token = True):
    attn_outputs = {}
    mlp_outputs = {}

    models_act = {}

    for couple in models:
        model = couple[0]
        model_name = couple[1]
        n_layers = model.cfg.n_layers


        model_cache = forward_with_cache(model)

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

        models_act[model_name] = {
            'mlp_act' : mlp_outputs, 
            'attn_act' : attn_outputs
        }

        attn_outputs = {}
        mlp_outputs = {}

    return models_act


import os 
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch as th

import models.rope_theta

from transformers import AutoConfig
from transformer_lens import HookedTransformer
from pathlib import Path

from models.model_wrapper import load_merged_model
from models.model import MODEL_ID, get_model, get_tokenizer

# seting models dir and device
device = 'cuda' if th.cuda.is_available() else 'cpu'

ADAPTER_DIR = 'adapters'
RLVR_ADAPTER_DIR = f'{ADAPTER_DIR}/rlvr_GMS8K_adapter'
SFTT_ADAPTER_DIR = f'{ADAPTER_DIR}/sftt_GMS8K_adapter'

base_tl = HookedTransformer.from_pretrained_no_processing(
    MODEL_ID,
    hf_model = get_model(),
    tokenizer = get_tokenizer(),
    device = device,
    dtype=th.bfloat16,
)
sff_tl = HookedTransformer.from_pretrained_no_processing(
    MODEL_ID,
    hf_model = load_merged_model(SFTT_ADAPTER_DIR),
    tokenizer = get_tokenizer(),
    device = device,
    dtype=th.bfloat16,
)
rlvr_tl = HookedTransformer.from_pretrained_no_processing(
    MODEL_ID,
    hf_model = load_merged_model(RLVR_ADAPTER_DIR),
    tokenizer = get_tokenizer(),
    device = device,
    dtype=th.bfloat16,
)

models = [
    (base_tl, 'base'), 
    (sff_tl, 'sftt'), 
    (rlvr_tl, 'rlvr')
]

# Setting system prompt
SYSTEM_PROMPT = 'Solve this math problem thinking step by step'

def forward_with_cache(model, prompts):
    with th.no_grad():
        _, cache = model.run_with_cache(prompts) # Forward with caching to extract representation

    return cache

def extract_mlp_attn_out(prompts, last_token = True):
    attn_outputs = {}
    mlp_outputs = {}

    models_act = {}

    for couple in models:
        model = couple[0]
        model_name = couple[1]
        n_layers = model.cfg.n_layers


        model_cache = forward_with_cache(prompts, model)

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

def extract_residual_out(prompts, last_token = True):
    
    model_residuals = {}

    for couple in models:
        resid_post_outs = {}
        resid_attn_outs = {}
        resid_attn_mlp_outs = {}

        model = couple[0]
        model_name = couple[1]

        model_cache = forward_with_cache(prompts, model)
        n_layers = model.cfg.n_layers
        
        interesting_layers = [0, n_layers // 2, n_layers - 1]

        for l in interesting_layers:
            if last_token:
                # Shape : [B, d_model]
                resid_post = model_cache['resid_post', l][:, -1, :].detach().cpu()
                resid_attn = model_cache['resid_mid', l][:, -1, :].detach().cpu()
                resid_attn_mlp = model_cache['resid_post', l][:, -1, :].detach().cpu()

            else :
                # Shape : [B, post, d_model]
                resid_post = model_cache['resid_post', l].detach().cpu()
                resid_attn = model_cache['resid_mid', l].detach().cpu()
                resid_attn_mlp = model_cache['resid_post', l].detach().cpu()

            resid_post_outs[l] = resid_post 
            resid_attn_outs[l] = resid_attn
            resid_attn_mlp_outs[l] = resid_attn_mlp
                        
            model_residuals[model_name] = {
                'resid_post_outs' : resid_post_outs,
                'resid_attn_outs' : resid_attn_outs, 
                'resid_attn_mlp_outs' : resid_attn_mlp_outs
            }

    return model_residuals
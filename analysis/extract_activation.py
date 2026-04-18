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

hf_base = get_model().to(device).eval()
hf_sftt = load_merged_model(SFTT_ADAPTER_DIR).to(device).eval()
hf_rlvr = load_merged_model(RLVR_ADAPTER_DIR).to(device).eval()

tokenizer = get_tokenizer()

base_tl = HookedTransformer.from_pretrained_no_processing(
    MODEL_ID,
    hf_model = hf_base,
    tokenizer = tokenizer,
    device = device,
    dtype=th.bfloat16,
)
sff_tl = HookedTransformer.from_pretrained_no_processing(
    MODEL_ID,
    hf_model = hf_sftt,
    tokenizer = tokenizer,
    device = device,
    dtype=th.bfloat16,
)
rlvr_tl = HookedTransformer.from_pretrained_no_processing(
    MODEL_ID,
    hf_model = hf_rlvr,
    tokenizer = tokenizer,
    device = device,
    dtype=th.bfloat16,
)

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

def extract_mlp_attn_out(prompts, last_token = True):
    attn_outputs = {}
    mlp_outputs = {}

    models_act = {}

    for couple in models:
        model = couple[0]
        model_name = couple[1]
        n_layers = model.cfg.n_layers

        model_cache = forward_with_cache(model, prompts, last_token)

        # Extract attn and mlp representation for each layers
        for l in range(n_layers):
            # Shape : [B, d_model] or [B, pos, d_mod]
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

    for model, model_name in models:
        n_layers = model.cfg.n_layers
        interesting_layers = [0, n_layers // 2, n_layers - 1]

        hook_names =[]
        for l in interesting_layers:
            hook_names.append(f'blocks.{l}.hook_resid_pre')
            hook_names.append(f'blocks.{l}.hook_resid_mid')
            hook_names.append(f'blocks.{l}.hook_resid_post')

        model_cache = forward_with_cache(model, prompts, last_token, hook_names)
        
        resid_pre_outs = {}
        resid_attn_outs = {}
        resid_attn_mlp_outs = {}

        for hook in hook_names:
            # Shape : [B, d_model] or [B, pos, d_model]
            resid_pre = model_cache[hook].detach().cpu()
            resid_attn = model_cache[hook].detach().cpu()
            resid_attn_mlp = model_cache[hook].detach().cpu()

            resid_pre_outs[l] = resid_pre 
            resid_attn_outs[l] = resid_attn
            resid_attn_mlp_outs[l] = resid_attn_mlp
                        
        model_residuals[model_name] = {
            'resid_pre_outs' : resid_pre_outs,
            'resid_attn_outs' : resid_attn_outs, 
            'resid_attn_mlp_outs' : resid_attn_mlp_outs
        }

    return model_residuals
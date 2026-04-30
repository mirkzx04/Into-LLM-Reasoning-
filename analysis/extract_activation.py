import os 
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch as th

import models.rope_theta

from transformers import AutoConfig
from transformer_lens import HookedTransformer
from tqdm import tqdm

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

def generate_rlvr_reasoning(prompts, max_new_tokens = 2000, do_sample = False, temperature = 0.7, top_p = 0.95, batch_size = 5):
    prompts_ids = tokenizer(
        prompts,
        return_tensor = "pt",
        add_special_token = False,
    ).inputs_ids.to(device)

    with th.inference_mode():
        full_ids = rlvr_tl.generate(
            prompts_ids,
            max_new_tokens=max_new_tokens,
            stop_at_eos=True,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            return_type="tokens",
            verbose=False
        )

    completion_ids = full_ids[:, prompts_ids.shape[-1]:]
    return prompts_ids, completion_ids, full_ids

def get_output_token_position(prompt_len, completion_len, mode):
    """
    mode = "state" 
        Take the positions of the completion tokens already present in the sequence

    mode = "predictive"
        takes positions that predict completion tokens

    mode = "prompt" 
        rakes the positions of input tokens
    """
    if mode == "state":
        start = prompt_len
        end = prompt_len + completion_len

    elif mode == "predictive":
        start = prompt_len - 1
        end = prompt_len + completion_len - 1
    elif mode == "prompt": 
        start = 0
        end = prompt_len

    return list(range(start, end))

def forward_with_cache(model, prompts, positions, names_filter = None):
    with th.no_grad():
        _, cache = model.run_with_cache(
            prompts,
            names_filter = names_filter,
            pos_slice = positions
        ) # Forward with caching to extract representation

    return cache

def get_hooks(interesting_layers, blocks):
    return [
        f"blocks.{l}.{b}"
        for l in interesting_layers
        for b in blocks
    ]

def extract_mlp_attn_out(prompts, interesting_layers, blocks, max_new_tokens, generation_do_sample = False, cache_mode = "state"):
    hook_names = get_hooks(interesting_layers, blocks)

    prompt_ids, completion_ids, full_ids = generate_rlvr_reasoning(
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        do_sample=generation_do_sample
    ) # Generate

    prompt_len = prompt_ids.shape[-1]
    completion_len = completion_ids.shape[-1]

    positions = get_output_token_position(
        prompt_len=prompt_len,
        completion_len=completion_len,
        mode = cache_mode
    )

    out = {
        "prompt": prompts,
        "prompt_ids": prompt_ids.detach().cpu(),
        "completion_ids": completion_ids.detach().cpu(),
        "full_ids": full_ids.detach().cpu(),
        "positions": positions,
        "cache_mode": cache_mode,
        "activations": {},
    }

    for model, model_name in models: 
        model_cache = forward_with_cache(
            model = model,
            input_ids = full_ids,
            positions=positions,
            names_filter=hook_names
        )

        layer_act = {}
        for l in interesting_layers:
            layer_act[l] = {
                block : model_cache[f"blocks.{l}.{block}"].detach().cpu()
                for block in blocks
            }
        out["activations"][model_name] = layer_act
    
        del model_cache
        th.cuda.empty_cache()

    return out

    
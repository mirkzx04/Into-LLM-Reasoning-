import os 
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import torch as th

from models.rope_theta import get_rope_theta

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

MODEL_ID = 'Qwen/Qwen2.5-1.5B'

def get_tokenizer(model_path = None):
    if not model_path: 
        model_path = MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Check padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    
    tokenizer.padding_side = 'left'
    return tokenizer

def get_hf_model(model_path = None):
    if not model_path: 
        model_path = MODEL_ID
    return AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype = th.bfloat16,
        attn_implementation = 'flash_attention_2',
    )

def load_tl_model(model_pth, device, n_ctx=None):
    # Build a TransformerLens model for one checkpoint variant.
    print(f"=== INSTANCE TL MODEL AS {model_pth} ===")
    hf_model = get_hf_model(model_pth)
    hf_tokenizer = get_tokenizer(model_pth)
    tl_model = HookedTransformer.from_pretrained_no_processing(
        MODEL_ID,
        hf_model=hf_model,
        tokenizer=hf_tokenizer,
        device=device,
        dtype=th.bfloat16,
        n_ctx=n_ctx,
    )

    del hf_model
    th.cuda.empty_cache()

    # Keep the config aligned with the runtime context window.
    tl_model.cfg.n_ctx = n_ctx
    tl_model.eval().to(device)
    return tl_model

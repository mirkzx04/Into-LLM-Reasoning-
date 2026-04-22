import os 
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import torch as th
from transformers import AutoModelForCausalLM, AutoTokenizer

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

def get_model(model_path = None):
    if not model_path: 
        model_path = MODEL_ID
    return AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype = th.bfloat16,
        attn_implementation = 'sdpa',
    )

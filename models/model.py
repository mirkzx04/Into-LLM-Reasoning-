import torch

from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = 'Qwen/Qwen2.5-3B'

def get_lora():
    return LoraConfig(
        r = 16, 
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj", 
            "v_proj", 
            "o_proj", 
            "gate_proj", 
            "up_proj", 
            "down_proj"
        ],
        task_type='CAUSAL_LM'
    )

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Check padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
    
    tokenizer.padding_side = 'left'
    return tokenizer

def get_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype = torch.bfloat16,
        attn_implementation = 'sdpa',
    )

import torch

from peft import LoraConfig
from transformers import AutoModelForCausalLM

def get_model():
    # Define LoRA instructions 
    peft_confing = LoraConfig(
        r = 16, 
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    MODEL_ID = 'Qwen/Qwen2.5-3B'
    model = AutoModelForCausalLM(
        MODEL_ID, 
        torch_dtype = torch.bfloat16,
        attn_implementation = 'flash_attention_2',
        device = 'auto'
    )

    return model, peft_confing
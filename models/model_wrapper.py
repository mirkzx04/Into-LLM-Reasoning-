from peft import PeftModel

from model import get_model

def gsm8k_rlvr_model(model_pth = 'rlvr_GMS8K_model'):
    model, lora_confg = get_model()
    model_rlvr = model.from_pretrained(model, model_pth)
    return model_rlvr.merge_and_unload(), lora_confg

def gsm8k_sftt_model(model_pth = 'sftt_GMS8K_model'):
    model, lora_confg = get_model()
    model_sftt = model.from_pretrained(model, model_pth)
    return model_sftt.merge_and_unload(), lora_confg

def vanilla_model():
    return get_model()
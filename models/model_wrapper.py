from peft import PeftModel

from models.model import get_model

def gsm8k_rlvr_model(model_pth='adapters/rlvr_GMS8K_adapter'):
    model = get_model()
    model_rlvr = PeftModel.from_pretrained(model, model_pth)
    return model_rlvr.merge_and_unload()

def gsm8k_sftt_model(model_pth='adapters/sftt_GMS8K_adapter'):
    model = get_model()
    model_sftt = PeftModel.from_pretrained(model, model_pth)
    return model_sftt.merge_and_unload()

def load_merged_model(adapter_dir):
    model = get_model()
    model = PeftModel.from_pretrained(model, adapter_dir)
    return model.merge_and_unload()

def export_merged_model(output_dir, adapter_dir):
    model = load_merged_model(adapter_dir)
    model.save_pretrained(output_dir)


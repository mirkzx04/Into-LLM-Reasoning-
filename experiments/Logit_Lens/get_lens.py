import os 
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import torch.nn as nn

from models.model import get_hf_model

def get_lens(model_pth, device):
    model = get_hf_model(model_pth)
    model.eval().to(device)

    # Extract tokens unmbedder 
    shared_norm = model.model.norm
    shared_lm_head = model.lm_head

    lens = nn.Sequential(
        shared_norm,
        shared_lm_head
    ).eval()

    # Setting requires grad = False for each parameters in lens
    for p in lens.parameters():
        p.requires_grad_(False)

    return lens
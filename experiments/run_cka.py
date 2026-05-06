import os 
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch as th
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from experiments.act_dataset_utils import get_activation_dataset, load_sample_batch

device = 'cuda' if th.cuda.is_available() else 'cpu'

def center_features(x) : 
    """
    Args :
        X L TorchTensor [B, d_model]
    """
    return x - x.mean(dim = 0, keepdim = True)

def linear_cka(x, y, eps = 1e-8):
    """
    Compute linear CKA between two feature metrics

    Args : 
        x, y : TorchTensor [B, d_model]
    """ 

    x = center_features(x)
    y = center_features(y)

    xxt = x.T @ x
    yyt = y.T @ y
    ytx = y.T @ x

    numerator = th.norm(ytx, p = 'fro') ** 2
    denominator = th.norm(xxt, p = 'fro') * th.norm(yyt, p = 'fro') + eps

    return (numerator / denominator).item()

dataset = get_activation_dataset()

load_sample_batch(
    batch_size=5,
    model_names="rlvr",
    act_modules = None,
    h5_path=dataset["h5_path"],
    layers=None
)
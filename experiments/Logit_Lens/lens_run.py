import os 
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

import torch as th

from itertools import combinations

from experiments.act_dataset_utils import load_sample_batch, get_activation_dataset

def create_model_comparison(model_names): 
    return list(combinations(model_names, 2))

# Get dataset path
act_dataset = get_activation_dataset()
h5_path = act_dataset["h5_path"]

act_modules = "resids"
slicing_mode = "position_normalized_completion"
normalized_pos = 0.1
layers = None
act_out, layer_ids = load_sample_batch(
    batch_size=10,
    model_names=None,
    act_modules=[act_modules],
    layers=layers,
    h5_path=h5_path,
    max_sample = 100
)
model_names = list(act_out.keys())

print("ciao")

import os 
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch as th
import numpy as np

from analysis.activation_view import choose_slicing

def build_cka_view(activation_out, model_names, act_name, layers, slicing_mode):
    reps = {}
    labels = []
    ordered_keys = []

    for model_name in model_names: 
        layer_buffer = [[] for _ in range(len(layers))] 
        sample_ids = sorted(activation_out[model_name][act_name].keys())

        for sid in sample_ids:
            act = activation_out[model_name][act_name][sid] # Shape : [L, seq_len, d_model]
            prompt_len = activation_out[model_name]["prompt_len"][sid]
            completion_len = activation_out[model_name]["completion_len"][sid]

            # Return the activation view based on slicing mode
            view = choose_slicing(
                slicing_mode=slicing_mode,
                act = act, 
                prompt_len=prompt_len,
                completion_len=completion_len
            )

            if view.ndim == 2: # [L, d_model]   
                for li in range(len(layers)):
                    layer_buffer[li].append(view[li][None, :])
            elif view.ndim == 3: # [L, T, D]
                for li in range(len(layers)):
                    layer_buffer[li].append(view[li])   # [T_i, D]

        for li, layer_name in enumerate(layers):
            X = np.concatenate(layer_buffer[li], axis=0).astype(np.float32)
            key = (model_name, layer_name)

            reps[key] = X
            ordered_keys.append(key)
            labels.append(f"{model_name}-{layer_name}")

    return reps, ordered_keys, labels

import time
import numpy as np
import torch as th


def inspect_cka_workload(reps, ordered_keys):
    print(f"num_reps = {len(ordered_keys)}")
    total_mem_mb = 0.0

    for key in ordered_keys:
        X = reps[key]

        if isinstance(X, np.ndarray):
            n, d = X.shape
            mem_mb = X.nbytes / (1024 ** 2)
        elif th.is_tensor(X):
            n, d = X.shape
            mem_mb = (X.numel() * X.element_size()) / (1024 ** 2)
        else:
            raise TypeError(f"Unsupported type for {key}: {type(X)}")

        total_mem_mb += mem_mb
        print(f"{key}: shape=({n}, {d}), mem={mem_mb:.2f} MB")

    n_reps = len(ordered_keys)
    n_pairs = n_reps * (n_reps + 1) // 2
    print(f"num_pairs = {n_pairs}")

    # stima grezza: per ogni coppia fai almeno un X^T Y con X,Y \in R^{N x D}
    # costo ~ N * D^2 multiply-add
    if len(ordered_keys) > 0:
        X0 = reps[ordered_keys[0]]
        n, d = X0.shape
        per_pair_ops = n * d * d
        total_ops = n_pairs * per_pair_ops

        print(f"rough_ops_per_pair ~ {per_pair_ops / 1e9:.2f}e9")
        print(f"rough_total_ops    ~ {total_ops / 1e9:.2f}e9")
        print(f"total_input_mem    ~ {total_mem_mb:.2f} MB")


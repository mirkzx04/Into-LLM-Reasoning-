import os 
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch as th
import matplotlib.pyplot as plt
import numpy as np

from analysis.extract_activation import extract_mlp_attn_out

device = 'cuda' if th.cuda.is_available() else 'cpu'

def center_features(x) : 
    """
    Args :
        X L TorchTensor [B, d_model]
    """
    return x - x.mena(dim = 0, keepdim = True)

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

def compute_pairwise_cka(model_acts):
    pairs = [
        ('base', 'sftt'),
        ('base', 'rlvr'),
        ('sftt', 'rlvr')
    ]
    components = ['attn_act', 'mlp_act']
    results = {comp : {} for comp in components}

    for comp in components:
        for m1, m2 in pairs:
            pair_name = f"{m1}_vs_{m2}"
            results[comp][pair_name] = {}

            layers = model_acts[m1][comp].keys()

            for l in layers:
                X = model_acts[m1][comp][l].float().to(device)
                Y = model_acts[m2][comp][l].float().to(device)

                if X.shape != Y.shape : 
                    raise ValueError(
                        f'Shape mismatch for {comp}, layer {l}, pair {pair_name}' 
                        f'{X.shape} vs {Y.shape}'
                    )

                # Check shape
                if X.ndim == 3:
                    B, P, D = X.shape
                    X = X.reshape(B * P, D)
                    Y = Y.reshape(B * P, D)
                
                cka_value = linear_cka(X, Y)
                results[comp][pair_name][l] = cka_value

                del X, Y

def plot_cka_by_layer(cka_results):
    for comp in ["attn_act", "mlp_act"]:
        plt.figure(figsize=(8, 5))

        for pair_name, layer_dict in cka_results[comp].items():
            layers = sorted(layer_dict.keys())
            values = [layer_dict[l] for l in layers]
            plt.plot(layers, values, marker="o", label=pair_name)

        plt.xlabel("Layer")
        plt.ylabel("CKA")
        plt.title(f"CKA by layer - {comp}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def plot_cka_heatmap(cka_results):
    for comp in ["attn_act", "mlp_act"]:
        pair_names = list(cka_results[comp].keys())
        layers = sorted(next(iter(cka_results[comp].values())).keys())

        matrix = []
        for pair_name in pair_names:
            row = [cka_results[comp][pair_name][l] for l in layers]
            matrix.append(row)

        matrix = np.array(matrix)

        plt.figure(figsize=(10, 4))
        plt.imshow(matrix, aspect="auto")
        plt.colorbar(label="CKA")
        plt.xticks(range(len(layers)), layers)
        plt.yticks(range(len(pair_names)), pair_names)
        plt.xlabel("Layer")
        plt.ylabel("Model pair")
        plt.title(f"CKA heatmap - {comp}")
        plt.tight_layout()
        plt.show()

def compute_cka(last_token = True):
    model_acts = extract_mlp_attn_out(prompts='', last_token = last_token)
    cka_results = compute_pairwise_cka(model_acts=model_acts)
    
    plot_cka_by_layer(cka_results)
    plot_cka_heatmap(cka_results)


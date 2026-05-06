
import os 
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np
import torch as th

from act_dataset_utils import load_sample_batch, get_activation_dataset
from CKA.cka_utils import build_cka_view, inspect_cka_workload
from CKA.compute_cka import compute_cka_matrix_profiled

import matplotlib.pyplot as plt

def plot_cka_heatmap(C, labels, title="CKA Heatmap"):
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(C, vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticklabels(labels)

    ax.set_title(title)

    # separatori tra modelli
    model_prefixes = [lab.split("-")[0] for lab in labels]
    for i in range(1, len(labels)):
        if model_prefixes[i] != model_prefixes[i - 1]:
            ax.axhline(i - 0.5, linewidth=2)
            ax.axvline(i - 0.5, linewidth=2)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Linear CKA")

    plt.tight_layout()
    plt.show()

from itertools import combinations


def print_same_layer_cka_markdown(
    C,
    ordered_keys,
    model_order=("base", "sftt", "rlvr"),
    decimals=4,
    title="Same-layer CKA"
):
    key_to_idx = {key: idx for idx, key in enumerate(ordered_keys)}

    present_models = []
    for model_name, _ in ordered_keys:
        if model_name not in present_models:
            present_models.append(model_name)

    ordered_models = [
        m for m in model_order if m in present_models
    ] + [
        m for m in present_models if m not in model_order
    ]

    layers = []
    for _, layer_name in ordered_keys:
        if layer_name not in layers:
            layers.append(layer_name)

    model_pairs = list(combinations(ordered_models, 2))

    print(f"\n### {title}\n")

    headers = ["Layer"] + [f"{m1}-{m2}" for m1, m2 in model_pairs]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] + ["---:"] * len(model_pairs)) + " |")

    pair_values = {pair: [] for pair in model_pairs}

    for layer_name in layers:
        row = [layer_name]

        for m1, m2 in model_pairs:
            key_1 = (m1, layer_name)
            key_2 = (m2, layer_name)

            if key_1 not in key_to_idx or key_2 not in key_to_idx:
                row.append("NA")
                continue

            i = key_to_idx[key_1]
            j = key_to_idx[key_2]

            value = float(C[i, j])
            pair_values[(m1, m2)].append(value)
            row.append(f"{value:.{decimals}f}")

        print("| " + " | ".join(row) + " |")

    mean_row = ["mean"]
    for pair in model_pairs:
        vals = pair_values[pair]
        if len(vals) == 0:
            mean_row.append("NA")
        else:
            mean_row.append(f"{np.mean(vals):.{decimals}f}")

    print("| " + " | ".join(mean_row) + " |")
    print()

act_dataset = get_activation_dataset()
h5_path = act_dataset["h5_path"]

act_modules = "attn_out_act"
slicing_mode = "last_input_token"
layers = None
activation_out, layers_ids = load_sample_batch(
    batch_size = 5, 
    model_names=None, 
    act_modules=[act_modules], 
    layers=layers,
    h5_path=h5_path,
    max_sample = 1000
)
models_in_activation_out = list(activation_out.keys())

print("=== BUILDING CKA VIEW ===")
reps, ordered_keys, labels = build_cka_view(
    activation_out=activation_out,
    model_names=models_in_activation_out,
    act_name=act_modules,
    layers = layers_ids,
    slicing_mode=slicing_mode
)

print("=== COMPUTE CKA MATRIX ===")
th.set_float32_matmul_precision("high")
inspect_cka_workload(reps, ordered_keys)
C = compute_cka_matrix_profiled(reps, ordered_keys, device = "cuda" if th.cuda.is_available() else "cpu")

print_same_layer_cka_markdown(
    C=C,
    ordered_keys=ordered_keys,
    title=f"{act_modules} - {slicing_mode}",
    decimals=4
)
plot_cka_heatmap(
    C, 
    labels,
    title = f"{act_modules} CKA - {slicing_mode}"
)
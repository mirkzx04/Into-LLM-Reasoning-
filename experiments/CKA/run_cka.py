
import os 
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import numpy as np
import torch as th

from act_dataset_utils import load_sample_batch, get_activation_dataset
from CKA.cka_utils import build_cka_view, inspect_cka_workload
from CKA.compute_cka import compute_cka_matrix_profiled

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CKA_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CKA_DIR, "..", ".."))
CKA_IMG_DIR = os.path.join(CKA_DIR, "cka_img")
DEFAULT_ACTIVATION_H5_PATH = os.path.join(
    PROJECT_ROOT,
    "activation_dataset",
    "qwen25_1.5b_rlvr_ood_eval_dataset_max_2000_acts.h5",
)
CKA_COLORMAP = "cividis"
CKA_VMIN = 0.85
CKA_VMAX = 1.0
CKA_COLORBAR_TICKS = (0.85, 0.90, 0.95, 0.98, 1.00)

HEATMAP_CONFIGS = (
    {
        "act_module": "attn_out_act",
        "slicing_mode": "predictive_completion_mean",
        "output_path": os.path.join(CKA_IMG_DIR, "act", "attn_out_pred_comp_mean.png"),
        "title": "attn_out_act CKA - predictive_completion_mean",
    },
    {
        "act_module": "attn_out_act",
        "slicing_mode": "position_normalized_completion",
        "slicing_kwargs": {"normalized_pos": 0.25},
        "output_path": os.path.join(CKA_IMG_DIR, "act", "attn_out_position_normalized_completion_025.png"),
        "title": "attn_out_act CKA - position_normalized_completion=0.25",
    },
    {
        "act_module": "mlp_out_act",
        "slicing_mode": "last_input_token",
        "output_path": os.path.join(CKA_IMG_DIR, "mlp", "mlp_out_last_inp_tok.png"),
        "title": "mlp_out_act CKA - last_input_token",
    },
    {
        "act_module": "mlp_out_act",
        "slicing_mode": "predictive_completion_mean",
        "output_path": os.path.join(CKA_IMG_DIR, "mlp", "mlp_out_pred_comp_mean.png"),
        "title": "mlp_out_act CKA - predictive_completion_mean",
    },
    {
        "act_module": "mlp_out_act",
        "slicing_mode": "position_normalized_completion",
        "slicing_kwargs": {"normalized_pos": 0.25},
        "output_path": os.path.join(CKA_IMG_DIR, "mlp", "mlp_norm_pos_025.png"),
        "title": "mlp_out_act CKA - position_normalized_completion=0.25",
    },
)


def ensure_parent_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def plot_cka_heatmap(C, labels, title="CKA Heatmap", output_path=None):
    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(10.8, 8.8))
    fig.patch.set_facecolor("#05070a")
    ax.set_facecolor("#090d12")

    cmap = plt.get_cmap(CKA_COLORMAP).copy()
    cmap.set_under("#111827")
    im = ax.imshow(C, vmin=CKA_VMIN, vmax=CKA_VMAX, cmap=cmap)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, color="#c9d1d9")
    ax.set_yticklabels(labels, color="#c9d1d9")

    ax.set_title(title, fontsize=14, fontweight="bold", color="#f4f4f5", pad=14)
    ax.tick_params(colors="#c9d1d9", labelsize=8)

    # separatori tra modelli
    model_prefixes = [lab.split("-")[0] for lab in labels]
    for i in range(1, len(labels)):
        if model_prefixes[i] != model_prefixes[i - 1]:
            ax.axhline(i - 0.5, color="#f4f4f5", linewidth=1.4, alpha=0.75)
            ax.axvline(i - 0.5, color="#f4f4f5", linewidth=1.4, alpha=0.75)

    for spine in ax.spines.values():
        spine.set_color("#2b2f36")

    for row_idx in range(C.shape[0]):
        for col_idx in range(C.shape[1]):
            value = float(C[row_idx, col_idx])
            text_color = "#05070a" if value >= 0.94 else "#f4f4f5"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.3f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=6.5,
            )

    cbar = plt.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        extend="min",
        ticks=CKA_COLORBAR_TICKS,
    )
    cbar.set_label("Linear CKA (focused scale)", color="#f4f4f5")
    cbar.ax.tick_params(colors="#c9d1d9", labelsize=8)
    cbar.outline.set_edgecolor("#2b2f36")

    plt.tight_layout()
    if output_path is None:
        plt.show()
        return None

    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path

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


def resolve_activation_h5_path():
    if os.path.exists(DEFAULT_ACTIVATION_H5_PATH):
        return DEFAULT_ACTIVATION_H5_PATH

    act_dataset = get_activation_dataset()
    return act_dataset["h5_path"]


def run_cka_heatmap(config, h5_path, layers=None, batch_size=5, max_sample=1000, sample_seed=42):
    act_module = config["act_module"]
    slicing_mode = config["slicing_mode"]
    slicing_kwargs = config.get("slicing_kwargs", {})

    activation_out, layers_ids = load_sample_batch(
        batch_size=batch_size,
        model_names=None,
        act_modules=[act_module],
        layers=layers,
        h5_path=h5_path,
        max_sample=max_sample,
        sample_seed=sample_seed,
    )
    models_in_activation_out = list(activation_out.keys())

    print(f"=== BUILDING CKA VIEW | {config['title']} ===")
    reps, ordered_keys, labels = build_cka_view(
        activation_out=activation_out,
        model_names=models_in_activation_out,
        act_name=act_module,
        layers=layers_ids,
        slicing_mode=slicing_mode,
        **slicing_kwargs,
    )

    print(f"=== COMPUTE CKA MATRIX | {config['title']} ===")
    inspect_cka_workload(reps, ordered_keys)
    C = compute_cka_matrix_profiled(
        reps,
        ordered_keys,
        device="cuda" if th.cuda.is_available() else "cpu",
    )

    print_same_layer_cka_markdown(
        C=C,
        ordered_keys=ordered_keys,
        title=config["title"],
        decimals=4,
    )
    saved_path = plot_cka_heatmap(
        C,
        labels,
        title=config["title"],
        output_path=config["output_path"],
    )
    print("Saved CKA heatmap to:", saved_path)
    return saved_path


def main():
    h5_path = resolve_activation_h5_path()

    th.set_float32_matmul_precision("high")

    saved_paths = []
    for config in HEATMAP_CONFIGS:
        saved_paths.append(run_cka_heatmap(config=config, h5_path=h5_path))

    print("Saved CKA heatmaps:")
    for path in saved_paths:
        print("-", path)


if __name__ == "__main__":
    main()

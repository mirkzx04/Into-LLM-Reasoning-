import argparse
import math
import os
import re
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")

import torch as th

ACTIVATION_PATCHING_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ACTIVATION_PATCHING_DIR, "..", ".."))
sys.path.append(PROJECT_ROOT)

from experiments.Logit_Lens.lens_plot import (
    aggregate_over_samples,
    metric_display_name,
    plot_metric_panels,
    sanitize_path_part,
)
from experiments.activation_patching.patching_metrics import (
    delta_kl_divergence,
    delta_target_in_topk_hit,
)
from experiments.experiments_conf import PatchConfig
from experiments.experiments_utils import (
    OUT_METADA_KEY,
    resolve_act_module_names,
)


PATCH_METRICS = (
    "recovery_score",
    "delta_target_in_topk_hit",
    "delta_kl_divergence",
)

PATCH_COMPONENT_DIRS = {
    "attn_out_act",
    "mlp_out_act",
    "attn_resid",
    "mlp_resid",
}


def layer_sort_key(layer_label):
    label = str(layer_label)
    match = re.search(r"(\d+)$", label)

    if match is None:
        return (1, label)

    return (0, int(match.group(1)))


def infer_patch_name(patch_out):
    patch_names = [
        key
        for key in patch_out.keys()
        if key != OUT_METADA_KEY and not str(key).startswith("__")
    ]

    if not patch_names:
        raise ValueError("patch_out does not contain a patching result key")

    if len(patch_names) > 1:
        raise ValueError(f"Expected one patching result key, found: {patch_names}")

    return patch_names[0]


def infer_model_names(patch_out, patch_name=None):
    metadata = patch_out.get(OUT_METADA_KEY, {})
    receiver_name = metadata.get("recipient_name")
    donor_name = metadata.get("donor_name")

    if receiver_name is not None and donor_name is not None:
        return str(receiver_name), str(donor_name)

    patch_name = patch_name or infer_patch_name(patch_out)
    match = re.fullmatch(r"recivient-(.+)_donor-(.+)", str(patch_name))

    if match is None:
        raise ValueError(
            "Cannot infer receiver/donor model names from metadata or patching key: "
            f"{patch_name}"
        )

    return match.group(1), match.group(2)


def infer_layers(patching_ref, metadata):
    metadata_layers = metadata.get("layer_labels")

    if metadata_layers:
        return [layer for layer in metadata_layers if layer in patching_ref]

    return sorted(patching_ref.keys(), key=layer_sort_key)


def infer_act_names(patching_ref, metadata, layers):
    metadata_act_names = metadata.get("patch_modules")

    if metadata_act_names:
        return [
            act_name
            for act_name in metadata_act_names
            if any(act_name in patching_ref.get(layer, {}) for layer in layers)
        ]

    act_names = []
    for layer in layers:
        for act_name in patching_ref.get(layer, {}).keys():
            if act_name not in act_names:
                act_names.append(act_name)

    return act_names


def infer_positions(patching_ref, metadata, layers, act_names):
    metadata_positions = metadata.get("positions")

    if metadata_positions:
        return sorted(
            metadata_positions,
            key=lambda value: float(value) if value is not None else -math.inf,
        )

    positions = []
    for layer in layers:
        for act_name in act_names:
            for position in patching_ref.get(layer, {}).get(act_name, {}).keys():
                if position not in positions:
                    positions.append(position)

    return sorted(
        positions,
        key=lambda value: float(value) if value is not None else -math.inf,
    )


def resolve_component_dir(act_name):
    try:
        component_dir = resolve_act_module_names(act_name)["saved"]
    except KeyError:
        component_dir = str(act_name)

    if component_dir not in PATCH_COMPONENT_DIRS:
        return sanitize_path_part(component_dir)

    return component_dir


def as_sample_tensor(values):
    if th.is_tensor(values):
        tensor = values.detach().cpu()
    elif isinstance(values, list):
        tensor = th.stack(
            [
                value.detach().cpu().reshape(())
                if th.is_tensor(value)
                else th.as_tensor(value).reshape(())
                for value in values
            ]
        )
    else:
        tensor = th.as_tensor(values)

    tensor = tensor.detach().cpu()

    while tensor.ndim > 1 and tensor.shape[-1] == 1:
        tensor = tensor.squeeze(-1)

    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)

    if tensor.ndim != 1:
        tensor = tensor.reshape(-1)

    if not th.is_floating_point(tensor):
        tensor = tensor.to(dtype=th.float32)

    return tensor


def extract_patch_metric(metrics_ref, metric_name):
    if metric_name == "recovery_score":
        return as_sample_tensor(metrics_ref["recovery_score"])

    if metric_name == "delta_target_in_topk_hit":
        return as_sample_tensor(
            delta_target_in_topk_hit(
                patched_hit=as_sample_tensor(metrics_ref["patched_hit"]),
                receiver_hit=as_sample_tensor(metrics_ref["recivier_hit"]),
            )
        )

    if metric_name == "delta_kl_divergence":
        return as_sample_tensor(
            delta_kl_divergence(
                kl_recivier_donor=as_sample_tensor(metrics_ref["kl_recivier_donor"]),
                kl_patched_donor=as_sample_tensor(metrics_ref["kl_patched_donor"]),
            )
        )

    raise ValueError(f"Unknown patch metric: {metric_name}")


def build_metric_tensor_for_position(
    patching_ref,
    layers,
    act_name,
    position,
    metric_name,
):
    layer_values = []

    for layer in layers:
        metrics_ref = patching_ref[layer][act_name][position]
        layer_values.append(
            extract_patch_metric(
                metrics_ref=metrics_ref,
                metric_name=metric_name,
            )
        )

    sample_counts = {values.shape[0] for values in layer_values}
    if len(sample_counts) != 1:
        raise ValueError(
            "Inconsistent sample counts across layers for "
            f"act_name={act_name}, position={position}, metric={metric_name}: "
            f"{sorted(sample_counts)}"
        )

    return th.stack(layer_values, dim=1)


def collect_patch_metric_series(
    patching_ref,
    metric_name,
    act_name,
    positions,
    layers,
    series_name,
    sample_reducer=None,
):
    series_by_position = {}

    for position in positions:
        try:
            metric_tensor = build_metric_tensor_for_position(
                patching_ref=patching_ref,
                layers=layers,
                act_name=act_name,
                position=position,
                metric_name=metric_name,
            )
        except KeyError:
            continue

        center, q1, q3 = aggregate_over_samples(
            metric_tensor=metric_tensor,
            sample_reducer=sample_reducer,
        )
        series_by_position[position] = {
            series_name: {
                "center": center,
                "q1": q1,
                "q3": q3,
            }
        }

    return series_by_position


def build_patch_plot_output_path(
    output_root,
    receiver_name,
    donor_name,
    act_name,
    metric_name,
):
    component_dir = resolve_component_dir(act_name)

    return os.path.join(
        output_root,
        f"{sanitize_path_part(receiver_name)}-{sanitize_path_part(donor_name)}",
        component_dir,
        f"{sanitize_path_part(metric_name)}.png",
    )


def resolve_default_patch_cache_path(config=None):
    config = config or PatchConfig()
    position_tag = "_".join(str(p).replace(".", "p") for p in config.positions)
    module_tag = "_".join(config.patch_modules)

    return os.path.join(
        ACTIVATION_PATCHING_DIR,
        "patch_cache",
        f"patch_out_recipient-{config.recipient_name}_"
        f"donor-{config.donor_name}_"
        f"modules-{module_tag}_"
        f"pos-{position_tag}.pt",
    )


def load_patch_out(patch_out_path):
    if not os.path.exists(patch_out_path):
        raise FileNotFoundError(f"patch_out cache not found: {patch_out_path}")

    return th.load(patch_out_path, map_location="cpu")


def plot_requested_patch_metrics(
    patch_out,
    output_root=None,
    metric_names=None,
    act_names=None,
    positions=None,
    layers=None,
    sample_reducer=None,
):
    output_root = output_root or ACTIVATION_PATCHING_DIR
    metric_names = metric_names or PATCH_METRICS

    patch_name = infer_patch_name(patch_out)
    patching_ref = patch_out[patch_name]
    metadata = patch_out.get(OUT_METADA_KEY, {})
    receiver_name, donor_name = infer_model_names(
        patch_out=patch_out,
        patch_name=patch_name,
    )

    layers = layers or infer_layers(
        patching_ref=patching_ref,
        metadata=metadata,
    )
    act_names = act_names or infer_act_names(
        patching_ref=patching_ref,
        metadata=metadata,
        layers=layers,
    )
    positions = positions or infer_positions(
        patching_ref=patching_ref,
        metadata=metadata,
        layers=layers,
        act_names=act_names,
    )

    series_name = f"{receiver_name}_vs_{donor_name}"
    saved_paths = []

    for act_name in act_names:
        for metric_name in metric_names:
            series_by_position = collect_patch_metric_series(
                patching_ref=patching_ref,
                metric_name=metric_name,
                act_name=act_name,
                positions=positions,
                layers=layers,
                series_name=series_name,
                sample_reducer=sample_reducer,
            )

            if not series_by_position:
                continue

            output_path = build_patch_plot_output_path(
                output_root=output_root,
                receiver_name=receiver_name,
                donor_name=donor_name,
                act_name=act_name,
                metric_name=metric_name,
            )
            title = (
                "Activation Patching | "
                f"{metric_display_name(metric_name)} | "
                f"{resolve_component_dir(act_name)} | "
                f"{receiver_name} <- {donor_name}"
            )
            saved_path = plot_metric_panels(
                series_by_position=series_by_position,
                metric_name=metric_name,
                output_path=output_path,
                title=title,
                layer_labels=layers,
            )

            if saved_path:
                saved_paths.append(saved_path)

    return saved_paths


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Generate activation patching plots from a patch_out cache."
    )
    parser.add_argument(
        "patch_out_path",
        nargs="?",
        default=None,
        help="Path to a patch_out .pt cache. Defaults to the PatchConfig cache path.",
    )
    parser.add_argument(
        "--output-root",
        default=ACTIVATION_PATCHING_DIR,
        help="Base directory where model/component plot folders are created.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    patch_out_path = args.patch_out_path or resolve_default_patch_cache_path()
    patch_out = load_patch_out(patch_out_path)
    saved_paths = plot_requested_patch_metrics(
        patch_out=patch_out,
        output_root=args.output_root,
    )

    print(f"Saved {len(saved_paths)} activation patching plot(s).")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()

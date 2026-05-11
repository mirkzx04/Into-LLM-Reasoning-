import hashlib
import math
import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-codex")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch as th

from experiments.Logit_Lens.lens_metrics_derived import (
    component_logprob_gain,
    delta_jaccard,
    delta_pairwise_DKL,
)


LOGIT_LENS_DIR = os.path.abspath(os.path.dirname(__file__))
DEFAULT_OUTPUT_ROOT = os.path.join(LOGIT_LENS_DIR, "logit_lens_img")
METADATA_KEY = "__metadata__"

PAIRWISE_METRICS = ("topk_jaccard", "kl_divergence")
COMPONENT_METRICS = (
    "delta_attn_logprob",
    "delta_mlp_logprob",
    "component_preference",
)
PAIRWISE_DKL_DELTA_METRIC = "delta_pairwise_DKL"
PAIRWISE_DKL_DELTA_COMPARISONS = (
    "base_vs_rlvr",
    "base_vs_sftt",
    "rlvr_vs_sftt",
)
PAIRWISE_JACCARD_DELTA_METRIC = "delta_jaccard"
REQUESTED_METRICS = PAIRWISE_METRICS + COMPONENT_METRICS
DISABLED_REPORT_METRICS = {
    "target_logprob",
    "target_rank",
    "top_ids",
}

MODEL_COLOR_PALETTE = (
    "#00f5ff",  # electric cyan
    "#ff2e88",  # hot magenta
    "#d6ff00",  # acid yellow
    "#7c5cff",
    "#00ff85",
    "#ff8a00",
)

MODEL_COLORS = {
    "base": "#00f5ff",
    "sftt": "#ff2e88",
    "rlvr": "#d6ff00",
    "base_vs_sftt": "#7c5cff",
    "sftt_vs_base": "#7c5cff",
    "base_vs_rlvr": "#00ff85",
    "rlvr_vs_base": "#00ff85",
    "sftt_vs_rlvr": "#ff8a00",
    "rlvr_vs_sftt": "#ff8a00",
}


def color_for_series(series_name):
    series_name = str(series_name)

    if series_name in MODEL_COLORS:
        return MODEL_COLORS[series_name]

    digest = hashlib.sha256(series_name.encode("utf-8")).digest()
    return MODEL_COLOR_PALETTE[digest[0] % len(MODEL_COLOR_PALETTE)]


def sanitize_path_part(value):
    """Return a filesystem-safe, readable path component."""
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "unnamed"


def is_metadata_key(key):
    return key == METADATA_KEY or str(key).startswith("__")


def infer_positions(lens_out):
    positions = [key for key in lens_out.keys() if not is_metadata_key(key)]
    return sorted(positions, key=lambda value: float(value) if value is not None else -math.inf)


def infer_act_names(lens_out, positions=None):
    positions = positions or infer_positions(lens_out)
    act_names = []

    for position in positions:
        for act_name in lens_out[position].keys():
            if act_name not in act_names:
                act_names.append(act_name)

    return act_names


def infer_lens_modes(lens_out, positions=None, act_names=None):
    positions = positions or infer_positions(lens_out)
    act_names = act_names or infer_act_names(lens_out, positions=positions)
    lens_modes = []

    for position in positions:
        for act_name in act_names:
            act_ref = lens_out.get(position, {}).get(act_name, {})
            for lens_mode in act_ref.keys():
                if lens_mode not in lens_modes:
                    lens_modes.append(lens_mode)

    return lens_modes


def infer_model_names(lens_out):
    for position in infer_positions(lens_out):
        for act_ref in lens_out[position].values():
            for lens_mode_ref in act_ref.values():
                model_names = []
                for series_name, metric_ref in lens_mode_ref.items():
                    if "_vs_" in str(series_name):
                        continue
                    if isinstance(metric_ref, dict) and "target_logprob" in metric_ref:
                        model_names.append(series_name)
                if model_names:
                    return model_names

    return []


def infer_comparison_names(lens_out):
    comparison_names = []

    for position in infer_positions(lens_out):
        for act_ref in lens_out[position].values():
            for lens_mode_ref in act_ref.values():
                for series_name, metric_ref in lens_mode_ref.items():
                    if not isinstance(metric_ref, dict):
                        continue

                    is_pairwise = "_vs_" in str(series_name)
                    has_pairwise_metric = any(metric in metric_ref for metric in PAIRWISE_METRICS)
                    if (is_pairwise or has_pairwise_metric) and series_name not in comparison_names:
                        comparison_names.append(series_name)

    return comparison_names


def as_sample_layer_tensor(metric_tensor):
    """Normalize metric values to a float tensor with shape [B, L]."""
    if not th.is_tensor(metric_tensor):
        metric_tensor = th.as_tensor(metric_tensor)

    metric_tensor = metric_tensor.detach().cpu()

    while metric_tensor.ndim > 2 and metric_tensor.shape[-1] == 1:
        metric_tensor = metric_tensor.squeeze(-1)

    if metric_tensor.ndim == 1:
        metric_tensor = metric_tensor.unsqueeze(0)

    if metric_tensor.ndim != 2:
        raise ValueError(
            f"Expected metric tensor with shape [B, L] or [L], got {tuple(metric_tensor.shape)}"
        )

    if not th.is_floating_point(metric_tensor):
        metric_tensor = metric_tensor.to(dtype=th.float32)

    return metric_tensor


def aggregate_over_samples(metric_tensor, sample_reducer=None):
    """Aggregate over samples with a median center and IQR band."""
    metric_tensor = as_sample_layer_tensor(metric_tensor)

    if sample_reducer is None:
        center = metric_tensor.quantile(0.5, dim=0)
    else:
        center = sample_reducer(metric_tensor)
        if not th.is_tensor(center):
            center = th.as_tensor(center)
        center = center.detach().cpu().to(dtype=th.float32)

    q1 = metric_tensor.quantile(0.25, dim=0)
    q3 = metric_tensor.quantile(0.75, dim=0)

    return center.numpy(), q1.numpy(), q3.numpy()


def build_comparison_plot_output_path(
    output_root,
    module_name=None,
    lens_mode=None,
    metric_name=None,
    act_name=None,
    suffix=None,
    **_,
):
    module_name = module_name or act_name

    filename_parts = [sanitize_path_part(metric_name)]
    if suffix:
        filename_parts.append(sanitize_path_part(suffix))
    filename = "__".join(filename_parts) + ".png"

    return os.path.join(
        output_root,
        sanitize_path_part(module_name),
        sanitize_path_part(lens_mode),
        filename,
    )


def build_metric_plot_output_path(
    output_root,
    module_name=None,
    lens_mode=None,
    metric_name=None,
    act_name=None,
    suffix=None,
):
    module_name = module_name or act_name
    filename_parts = [sanitize_path_part(metric_name)]

    if suffix:
        filename_parts.append(sanitize_path_part(suffix))

    return os.path.join(
        output_root,
        sanitize_path_part(module_name),
        sanitize_path_part(lens_mode),
        "__".join(filename_parts) + ".png",
    )


def build_pairwise_metric_plot_output_path(
    output_root,
    module_name=None,
    lens_mode=None,
    metric_name=None,
    act_name=None,
    **_,
):
    return build_metric_plot_output_path(
        output_root=output_root,
        module_name=module_name,
        act_name=act_name,
        lens_mode=lens_mode,
        metric_name=metric_name,
    )


def build_component_metric_plot_output_path(
    output_root,
    lens_mode=None,
    metric_name=None,
    module_name="component_logprob",
    suffix=None,
    **_,
):
    return build_metric_plot_output_path(
        output_root=output_root,
        module_name=module_name,
        lens_mode=lens_mode,
        metric_name=metric_name,
        suffix=suffix,
    )


def build_component_delta_table_output_path(
    output_root,
    lens_mode=None,
    model_name=None,
    module_name="component_logprob",
):
    return os.path.join(
        output_root,
        sanitize_path_part(module_name),
        sanitize_path_part(lens_mode),
        "tables",
        f"{sanitize_path_part(model_name)}__delta_table.png",
    )


def build_pairwise_dkl_delta_plot_output_path(
    output_root,
    lens_mode=None,
    metric_name=PAIRWISE_DKL_DELTA_METRIC,
    module_name="pairwise_dkl_delta",
):
    return build_metric_plot_output_path(
        output_root=output_root,
        module_name=module_name,
        lens_mode=lens_mode,
        metric_name=metric_name,
    )


def build_pairwise_jaccard_delta_plot_output_path(
    output_root,
    lens_mode=None,
    metric_name=PAIRWISE_JACCARD_DELTA_METRIC,
    module_name="pairwise_jaccard_delta",
):
    return build_metric_plot_output_path(
        output_root=output_root,
        module_name=module_name,
        lens_mode=lens_mode,
        metric_name=metric_name,
    )


def build_pairwise_plot_output_path(
    output_root,
    act_name=None,
    module_name=None,
    lens_mode=None,
    metric_name=None,
    **_,
):
    return build_pairwise_metric_plot_output_path(
        output_root=output_root,
        module_name=module_name or act_name,
        lens_mode=lens_mode,
        metric_name=metric_name,
    )


def ensure_parent_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_figure(fig, output_path, dpi=180):
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def metric_display_name(metric_name):
    labels = {
        "topk_jaccard": "Top-k Jaccard",
        "kl_divergence": "Symmetric DKL",
        "delta_attn_logprob": "Delta Attn LogProb",
        "delta_mlp_logprob": "Delta MLP LogProb",
        "component_preference": "Component Preference",
        "delta_pairwise_DKL": "DKL(MLP Resid) - DKL(Attn Resid)",
        "delta_jaccard": "Jaccard(MLP Resid) - Jaccard(Attn Resid)",
    }
    return labels.get(metric_name, str(metric_name).replace("_", " ").title())


def format_position(position):
    if position is None:
        return "fallback"
    return f"{float(position):.2f}"


def resolve_layer_axis(n_layers, layer_labels=None):
    x = list(range(n_layers))

    if layer_labels is None or len(layer_labels) != n_layers:
        return x, [str(item) for item in x]

    return x, [str(label) for label in layer_labels]


def style_axis(ax, x, x_labels, metric_name, position):
    ax.set_title(f"position={format_position(position)}", fontsize=10, color="#f4f4f5")
    ax.set_xlabel("Layer")
    ax.set_ylabel(metric_display_name(metric_name))
    ax.grid(True, color="#2b2f36", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#c9d1d9", labelsize=8)

    if metric_name in COMPONENT_METRICS or str(metric_name).startswith("delta_"):
        ax.axhline(0, color="#f4f4f5", linewidth=1.0, linestyle="--", alpha=0.65)

    if metric_name == "kl_divergence":
        ax.set_ylim(bottom=0)

    if len(x_labels) <= 24:
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
    else:
        step = max(1, len(x_labels) // 12)
        shown_x = x[::step]
        ax.set_xticks(shown_x)
        ax.set_xticklabels([x_labels[idx] for idx in shown_x], rotation=45, ha="right")


def plot_metric_panels(
    series_by_position,
    metric_name,
    output_path,
    title,
    layer_labels=None,
):
    positions = list(series_by_position.keys())
    if not positions:
        return None

    plt.style.use("dark_background")

    ncols = min(2, len(positions))
    nrows = math.ceil(len(positions) / ncols)
    fig_width = 7.2 * ncols
    fig_height = 4.6 * nrows
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharey=True,
    )
    axes_flat = axes.flatten()

    fig.patch.set_facecolor("#05070a")
    fig.suptitle(title, fontsize=14, fontweight="bold", color="#f4f4f5")

    legend_handles = []
    legend_labels = []

    for panel_idx, position in enumerate(positions):
        ax = axes_flat[panel_idx]
        ax.set_facecolor("#090d12")

        series_stats = series_by_position[position]
        first_stats = next(iter(series_stats.values()))
        x, x_labels = resolve_layer_axis(len(first_stats["center"]), layer_labels=layer_labels)

        for series_name, stats in series_stats.items():
            color = color_for_series(series_name)
            center = stats["center"]
            q1 = stats["q1"]
            q3 = stats["q3"]

            (line,) = ax.plot(
                x,
                center,
                color=color,
                linewidth=2.1,
                label=str(series_name),
            )
            ax.fill_between(
                x,
                q1,
                q3,
                color=color,
                alpha=0.16,
                linewidth=0,
            )

            if str(series_name) not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(str(series_name))

        style_axis(
            ax=ax,
            x=x,
            x_labels=x_labels,
            metric_name=metric_name,
            position=position,
        )

    for ax in axes_flat[len(positions):]:
        ax.axis("off")

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=min(3, len(legend_labels)),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    return save_figure(fig=fig, output_path=output_path)


def collect_direct_metric_series(
    lens_out,
    metric_name,
    act_name,
    lens_mode,
    positions,
    series_names,
    sample_reducer=None,
):
    series_by_position = {}

    for position in positions:
        lens_mode_ref = lens_out.get(position, {}).get(act_name, {}).get(lens_mode, {})
        panel_series = {}

        for series_name in series_names:
            metric_ref = lens_mode_ref.get(series_name, {})
            if metric_name not in metric_ref:
                continue

            center, q1, q3 = aggregate_over_samples(
                metric_ref[metric_name],
                sample_reducer=sample_reducer,
            )
            panel_series[series_name] = {"center": center, "q1": q1, "q3": q3}

        if panel_series:
            series_by_position[position] = panel_series

    return series_by_position


def collect_component_metric_series(
    lens_out,
    metric_name,
    lens_mode,
    positions,
    model_names,
    sample_reducer=None,
):
    series_by_position = {}

    for position in positions:
        panel_series = {}

        for model_name in model_names:
            try:
                gains = component_logprob_gain(
                    lens_out=lens_out,
                    position=position,
                    lens_mode=lens_mode,
                    model_name=model_name,
                )
            except KeyError:
                continue

            if metric_name == "component_preference":
                metric_tensor = gains["delta_mlp_logprob"] - gains["delta_attn_logprob"]
            else:
                metric_tensor = gains[metric_name]

            center, q1, q3 = aggregate_over_samples(
                metric_tensor,
                sample_reducer=sample_reducer,
            )
            panel_series[model_name] = {"center": center, "q1": q1, "q3": q3}

        if panel_series:
            series_by_position[position] = panel_series

    return series_by_position


def collect_pairwise_dkl_delta_series(
    lens_out,
    lens_mode,
    positions,
    comparison_names=None,
    sample_reducer=None,
):
    comparison_names = comparison_names or PAIRWISE_DKL_DELTA_COMPARISONS
    series_by_position = {}

    for position in positions:
        panel_series = {}

        for comparison_name in comparison_names:
            try:
                metric_tensor = delta_pairwise_DKL(
                    lens_out=lens_out,
                    position=position,
                    lens_mode=lens_mode,
                    comparison_name=comparison_name,
                )
            except KeyError:
                continue

            center, q1, q3 = aggregate_over_samples(
                metric_tensor,
                sample_reducer=sample_reducer,
            )
            panel_series[comparison_name] = {"center": center, "q1": q1, "q3": q3}

        if panel_series:
            series_by_position[position] = panel_series

    return series_by_position


def collect_pairwise_jaccard_delta_series(
    lens_out,
    lens_mode,
    positions,
    comparison_names=None,
    sample_reducer=None,
):
    comparison_names = comparison_names or PAIRWISE_DKL_DELTA_COMPARISONS
    series_by_position = {}

    for position in positions:
        panel_series = {}

        for comparison_name in comparison_names:
            try:
                metric_tensor = delta_jaccard(
                    lens_out=lens_out,
                    position=position,
                    lens_mode=lens_mode,
                    comparison_name=comparison_name,
                )
            except KeyError:
                continue

            center, q1, q3 = aggregate_over_samples(
                metric_tensor,
                sample_reducer=sample_reducer,
            )
            panel_series[comparison_name] = {"center": center, "q1": q1, "q3": q3}

        if panel_series:
            series_by_position[position] = panel_series

    return series_by_position


def is_layer_zero_label(label):
    label = str(label)
    match = re.search(r"(\d+)$", label)

    if match is None:
        return label == "0"

    return int(match.group(1)) == 0


def infer_series_n_layers(series_by_position):
    for panel_series in series_by_position.values():
        for stats in panel_series.values():
            return len(stats["center"])

    return 0


def resolve_without_layer0_selection(n_layers, layer_labels=None):
    if n_layers <= 1:
        return None, None

    if layer_labels is not None and len(layer_labels) == n_layers:
        keep_indices = [
            idx
            for idx, label in enumerate(layer_labels)
            if not is_layer_zero_label(label)
        ]

        if len(keep_indices) == n_layers:
            keep_indices = list(range(1, n_layers))

        if not keep_indices:
            return None, None

        return keep_indices, [layer_labels[idx] for idx in keep_indices]

    return list(range(1, n_layers)), None


def subset_series_layers(series_by_position, keep_indices):
    subset = {}

    for position, panel_series in series_by_position.items():
        subset[position] = {}

        for series_name, stats in panel_series.items():
            subset[position][series_name] = {
                key: value[keep_indices]
                for key, value in stats.items()
            }

    return subset


def format_table_float(value, decimals=6):
    return f"{float(value):.{decimals}f}"


def build_component_delta_table_rows(
    lens_out,
    lens_mode,
    model_name,
    positions,
    layer_labels=None,
    sample_reducer=None,
):
    sections = []

    for position in positions:
        try:
            gains = component_logprob_gain(
                lens_out=lens_out,
                position=position,
                lens_mode=lens_mode,
                model_name=model_name,
            )
        except KeyError:
            continue

        delta_attn, _, _ = aggregate_over_samples(
            gains["delta_attn_logprob"],
            sample_reducer=sample_reducer,
        )
        delta_mlp, _, _ = aggregate_over_samples(
            gains["delta_mlp_logprob"],
            sample_reducer=sample_reducer,
        )
        _, resolved_layer_labels = resolve_layer_axis(
            len(delta_attn),
            layer_labels=layer_labels,
        )

        rows = []
        for layer_label, delta_attn_value, delta_mlp_value in zip(
            resolved_layer_labels,
            delta_attn,
            delta_mlp,
        ):
            rows.append(
                [
                    str(layer_label),
                    format_table_float(delta_attn_value),
                    format_table_float(delta_mlp_value),
                ]
            )
        sections.append((format_position(position), rows))

    return sections


def style_delta_table(table):
    header_color = "#161a20"
    body_colors = ("#090d12", "#0c1117")
    edge_color = "#2b2f36"
    text_color = "#f4f4f5"

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(edge_color)
        cell.set_linewidth(0.7)
        cell.PAD = 0.08

        if row_idx == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color="#f4f4f5", weight="bold")
            continue

        cell.set_facecolor(body_colors[(row_idx - 1) % len(body_colors)])
        cell.set_text_props(color=text_color)


def plot_component_delta_table(
    sections,
    model_name,
    lens_mode,
    output_path,
):
    if not sections:
        return None

    plt.style.use("dark_background")

    ncols = min(2, len(sections))
    nrows = math.ceil(len(sections) / ncols)
    fig_width = 6.2 * ncols
    fig_height = 3.1 * nrows + 0.6
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )
    axes_flat = axes.flatten()
    fig.patch.set_facecolor("#05070a")

    fig.suptitle(
        f"Component Delta Table | {model_name} | {lens_mode}",
        fontsize=14,
        fontweight="bold",
        color="#f4f4f5",
    )

    for ax_idx, (position_label, rows) in enumerate(sections):
        ax = axes_flat[ax_idx]
        ax.set_facecolor("#05070a")
        ax.axis("off")
        ax.set_title(
            f"position={position_label}",
            fontsize=10,
            color="#d8dee9",
            pad=8,
        )

        table = ax.table(
            cellText=rows,
            colLabels=["layer", "delta_attn", "delta_mlp"],
            cellLoc="center",
            colLoc="center",
            loc="center",
            bbox=(0.04, 0.04, 0.92, 0.82),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.12)
        style_delta_table(table=table)

    for ax in axes_flat[len(sections):]:
        ax.axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.93))

    return save_figure(fig=fig, output_path=output_path)


def plot_component_delta_tables(
    lens_out,
    output_root=None,
    layer_labels=None,
    positions=None,
    lens_modes=None,
    model_names=None,
    sample_reducer=None,
    module_name="component_logprob",
):
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    positions = positions or infer_positions(lens_out)
    lens_modes = lens_modes or infer_lens_modes(lens_out, positions=positions)
    model_names = model_names or infer_model_names(lens_out)
    saved_paths = []

    for lens_mode in lens_modes:
        for model_name in model_names:
            sections = build_component_delta_table_rows(
                lens_out=lens_out,
                lens_mode=lens_mode,
                model_name=model_name,
                positions=positions,
                layer_labels=layer_labels,
                sample_reducer=sample_reducer,
            )
            if not sections:
                continue

            output_path = build_component_delta_table_output_path(
                output_root=output_root,
                lens_mode=lens_mode,
                model_name=model_name,
                module_name=module_name,
            )
            saved_path = plot_component_delta_table(
                sections=sections,
                model_name=model_name,
                lens_mode=lens_mode,
                output_path=output_path,
            )
            if saved_path:
                saved_paths.append(saved_path)

    return saved_paths


def plot_alignment_metric_panels(
    lens_out,
    metric_names,
    sample_reducer=None,
    output_root=None,
    lens_source_name=None,
    layer_labels=None,
    positions=None,
    act_names=None,
    model_names=None,
    output_path_builder=None,
    title_prefix="Logit Lens",
):
    """Plot direct metrics stored in lens_out, one comparison figure per metric."""
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    positions = positions or infer_positions(lens_out)
    act_names = act_names or infer_act_names(lens_out, positions=positions)
    lens_modes = infer_lens_modes(lens_out, positions=positions, act_names=act_names)
    model_names = model_names or infer_model_names(lens_out)
    output_path_builder = output_path_builder or build_comparison_plot_output_path
    saved_paths = []

    for act_name in act_names:
        for lens_mode in lens_modes:
            for metric_name in metric_names:
                if metric_name in DISABLED_REPORT_METRICS:
                    continue

                series_by_position = collect_direct_metric_series(
                    lens_out=lens_out,
                    metric_name=metric_name,
                    act_name=act_name,
                    lens_mode=lens_mode,
                    positions=positions,
                    series_names=model_names,
                    sample_reducer=sample_reducer,
                )

                if not series_by_position:
                    continue

                output_path = output_path_builder(
                    output_root=output_root,
                    act_name=act_name,
                    module_name=act_name,
                    lens_mode=lens_mode,
                    metric_name=metric_name,
                    lens_source_name=lens_source_name,
                )
                title = (
                    f"{title_prefix} | {metric_display_name(metric_name)} | "
                    f"{act_name} | {lens_mode}"
                )
                saved_path = plot_metric_panels(
                    series_by_position=series_by_position,
                    metric_name=metric_name,
                    output_path=output_path,
                    title=title,
                    layer_labels=layer_labels,
                )
                if saved_path:
                    saved_paths.append(saved_path)

    return saved_paths


def plot_pairwise_metrics(
    lens_out,
    output_root=None,
    layer_labels=None,
    positions=None,
    act_names=None,
    lens_modes=None,
    comparison_names=None,
    sample_reducer=None,
):
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    positions = positions or infer_positions(lens_out)
    act_names = act_names or infer_act_names(lens_out, positions=positions)
    lens_modes = lens_modes or infer_lens_modes(lens_out, positions=positions, act_names=act_names)
    comparison_names = comparison_names or infer_comparison_names(lens_out)
    saved_paths = []

    for act_name in act_names:
        for lens_mode in lens_modes:
            for metric_name in PAIRWISE_METRICS:
                series_by_position = collect_direct_metric_series(
                    lens_out=lens_out,
                    metric_name=metric_name,
                    act_name=act_name,
                    lens_mode=lens_mode,
                    positions=positions,
                    series_names=comparison_names,
                    sample_reducer=sample_reducer,
                )

                if not series_by_position:
                    continue

                output_path = build_pairwise_metric_plot_output_path(
                    output_root=output_root,
                    module_name=act_name,
                    lens_mode=lens_mode,
                    metric_name=metric_name,
                )
                saved_path = plot_metric_panels(
                    series_by_position=series_by_position,
                    metric_name=metric_name,
                    output_path=output_path,
                    title=f"Pairwise Model Comparison | {metric_display_name(metric_name)} | {act_name} | {lens_mode}",
                    layer_labels=layer_labels,
                )
                if saved_path:
                    saved_paths.append(saved_path)

    return saved_paths


def plot_pairwise_dkl_delta(
    lens_out,
    output_root=None,
    layer_labels=None,
    positions=None,
    lens_modes=None,
    comparison_names=None,
    sample_reducer=None,
    module_name="pairwise_dkl_delta",
):
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    positions = positions or infer_positions(lens_out)
    lens_modes = lens_modes or infer_lens_modes(
        lens_out,
        positions=positions,
        act_names=("attn_resid", "mlp_resid"),
    )
    comparison_names = comparison_names or PAIRWISE_DKL_DELTA_COMPARISONS
    saved_paths = []

    for lens_mode in lens_modes:
        series_by_position = collect_pairwise_dkl_delta_series(
            lens_out=lens_out,
            lens_mode=lens_mode,
            positions=positions,
            comparison_names=comparison_names,
            sample_reducer=sample_reducer,
        )

        if not series_by_position:
            continue

        output_path = build_pairwise_dkl_delta_plot_output_path(
            output_root=output_root,
            module_name=module_name,
            lens_mode=lens_mode,
            metric_name=PAIRWISE_DKL_DELTA_METRIC,
        )
        saved_path = plot_metric_panels(
            series_by_position=series_by_position,
            metric_name=PAIRWISE_DKL_DELTA_METRIC,
            output_path=output_path,
            title=(
                "Pairwise DKL Delta | "
                f"{metric_display_name(PAIRWISE_DKL_DELTA_METRIC)} | {lens_mode}"
            ),
            layer_labels=layer_labels,
        )
        if saved_path:
            saved_paths.append(saved_path)

    return saved_paths


def plot_pairwise_jaccard_delta(
    lens_out,
    output_root=None,
    layer_labels=None,
    positions=None,
    lens_modes=None,
    comparison_names=None,
    sample_reducer=None,
    module_name="pairwise_jaccard_delta",
):
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    positions = positions or infer_positions(lens_out)
    lens_modes = lens_modes or infer_lens_modes(
        lens_out,
        positions=positions,
        act_names=("attn_resid", "mlp_resid"),
    )
    comparison_names = comparison_names or PAIRWISE_DKL_DELTA_COMPARISONS
    saved_paths = []

    for lens_mode in lens_modes:
        series_by_position = collect_pairwise_jaccard_delta_series(
            lens_out=lens_out,
            lens_mode=lens_mode,
            positions=positions,
            comparison_names=comparison_names,
            sample_reducer=sample_reducer,
        )

        if not series_by_position:
            continue

        output_path = build_pairwise_jaccard_delta_plot_output_path(
            output_root=output_root,
            module_name=module_name,
            lens_mode=lens_mode,
            metric_name=PAIRWISE_JACCARD_DELTA_METRIC,
        )
        saved_path = plot_metric_panels(
            series_by_position=series_by_position,
            metric_name=PAIRWISE_JACCARD_DELTA_METRIC,
            output_path=output_path,
            title=(
                "Pairwise Jaccard Delta | "
                f"{metric_display_name(PAIRWISE_JACCARD_DELTA_METRIC)} | {lens_mode}"
            ),
            layer_labels=layer_labels,
        )
        if saved_path:
            saved_paths.append(saved_path)

    return saved_paths


def plot_component_metrics(
    lens_out,
    output_root=None,
    layer_labels=None,
    positions=None,
    lens_modes=None,
    model_names=None,
    sample_reducer=None,
    module_name="component_logprob",
):
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    positions = positions or infer_positions(lens_out)
    lens_modes = lens_modes or infer_lens_modes(lens_out, positions=positions)
    model_names = model_names or infer_model_names(lens_out)
    saved_paths = []

    for lens_mode in lens_modes:
        for metric_name in COMPONENT_METRICS:
            series_by_position = collect_component_metric_series(
                lens_out=lens_out,
                metric_name=metric_name,
                lens_mode=lens_mode,
                positions=positions,
                model_names=model_names,
                sample_reducer=sample_reducer,
            )

            if not series_by_position:
                continue

            output_path = build_component_metric_plot_output_path(
                output_root=output_root,
                module_name=module_name,
                lens_mode=lens_mode,
                metric_name=metric_name,
            )
            saved_path = plot_metric_panels(
                series_by_position=series_by_position,
                metric_name=metric_name,
                output_path=output_path,
                title=f"Component LogProb Dynamics | {metric_display_name(metric_name)} | {lens_mode}",
                layer_labels=layer_labels,
            )
            if saved_path:
                saved_paths.append(saved_path)

            n_layers = infer_series_n_layers(series_by_position)
            keep_indices, zoom_layer_labels = resolve_without_layer0_selection(
                n_layers=n_layers,
                layer_labels=layer_labels,
            )
            if keep_indices is None:
                continue

            zoom_series_by_position = subset_series_layers(
                series_by_position=series_by_position,
                keep_indices=keep_indices,
            )
            zoom_output_path = build_component_metric_plot_output_path(
                output_root=output_root,
                module_name=module_name,
                lens_mode=lens_mode,
                metric_name=metric_name,
                suffix="without_layer0",
            )
            zoom_saved_path = plot_metric_panels(
                series_by_position=zoom_series_by_position,
                metric_name=metric_name,
                output_path=zoom_output_path,
                title=(
                    "Component LogProb Dynamics | "
                    f"{metric_display_name(metric_name)} | {lens_mode} | without layer 0"
                ),
                layer_labels=zoom_layer_labels,
            )
            if zoom_saved_path:
                saved_paths.append(zoom_saved_path)

    return saved_paths


def plot_requested_logit_lens_metrics(
    lens_out,
    output_root=None,
    layer_labels=None,
    positions=None,
    act_names=None,
    lens_modes=None,
    model_names=None,
    comparison_names=None,
    sample_reducer=None,
):
    """Build the complete report requested for Logit Lens comparison plots."""
    output_root = output_root or DEFAULT_OUTPUT_ROOT
    positions = positions or infer_positions(lens_out)
    act_names = act_names or infer_act_names(lens_out, positions=positions)
    lens_modes = lens_modes or infer_lens_modes(lens_out, positions=positions, act_names=act_names)
    model_names = model_names or infer_model_names(lens_out)
    comparison_names = comparison_names or infer_comparison_names(lens_out)

    saved_paths = []
    saved_paths.extend(
        plot_pairwise_metrics(
            lens_out=lens_out,
            output_root=output_root,
            layer_labels=layer_labels,
            positions=positions,
            act_names=act_names,
            lens_modes=lens_modes,
            comparison_names=comparison_names,
            sample_reducer=sample_reducer,
        )
    )
    saved_paths.extend(
        plot_pairwise_dkl_delta(
            lens_out=lens_out,
            output_root=output_root,
            layer_labels=layer_labels,
            positions=positions,
            lens_modes=lens_modes,
            comparison_names=PAIRWISE_DKL_DELTA_COMPARISONS,
            sample_reducer=sample_reducer,
        )
    )
    saved_paths.extend(
        plot_pairwise_jaccard_delta(
            lens_out=lens_out,
            output_root=output_root,
            layer_labels=layer_labels,
            positions=positions,
            lens_modes=lens_modes,
            comparison_names=PAIRWISE_DKL_DELTA_COMPARISONS,
            sample_reducer=sample_reducer,
        )
    )
    saved_paths.extend(
        plot_component_metrics(
            lens_out=lens_out,
            output_root=output_root,
            layer_labels=layer_labels,
            positions=positions,
            lens_modes=lens_modes,
            model_names=model_names,
            sample_reducer=sample_reducer,
        )
    )

    return saved_paths

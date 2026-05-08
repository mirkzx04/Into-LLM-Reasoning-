import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch as th


ACT_DISPLAY_NAMES = {
    "resid_pre": "Residual Stream (Pre)",
    "attn_resid": "Attention Residual",
    "mlp_resid": "MLP Residual",
}

ACT_FILE_STEMS = {
    "resid_pre": "resid",
    "attn_resid": "attn",
    "mlp_resid": "mlp",
}

METRIC_DISPLAY_NAMES = {
    "entropy": "Entropy",
    "eff_vocab": "Effective Vocabulary",
    "top1_prob": "Top-1 Probability",
    "prob_margin": "Probability Margin",
    "logit_margin": "Logit Margin",
    "target_logprob": "Target Log-Probability",
    "target_prob": "Target Probability",
    "surprisal": "Surprisal",
    "target_rank": "Target Rank",
}


def build_lens_variant_dirname(lens_mode, lens_source_name=None):
    if lens_mode == "shared" and lens_source_name is not None:
        return f"shared_{lens_source_name}"

    return lens_mode


def build_lens_display_label(lens_mode, lens_source_name=None):
    if lens_mode == "shared" and lens_source_name is not None:
        return f"Shared Lens ({lens_source_name.upper()})"

    return f"{lens_mode.title()} Lens"


def infer_positions(lens_out):
    return sorted(key for key in lens_out.keys() if isinstance(key, (int, float)))


def infer_act_names(lens_out, positions=None):
    positions = infer_positions(lens_out) if positions is None else positions
    return list(lens_out[positions[0]].keys())


def infer_lens_modes(lens_out, positions=None, act_names=None):
    positions = infer_positions(lens_out) if positions is None else positions
    act_names = infer_act_names(lens_out, positions) if act_names is None else act_names
    return list(lens_out[positions[0]][act_names[0]].keys())


def infer_model_names(lens_out, positions=None, act_name=None, lens_mode=None):
    positions = infer_positions(lens_out) if positions is None else positions
    act_names = infer_act_names(lens_out, positions)
    act_name = act_names[0] if act_name is None else act_name
    lens_modes = infer_lens_modes(lens_out, positions, [act_name])
    lens_mode = lens_modes[0] if lens_mode is None else lens_mode

    ordered_names = []
    for key in lens_out[positions[0]][act_name][lens_mode].keys():
        if "_vs_" not in key:
            ordered_names.append(key)

    return ordered_names


def build_layer_ticks(n_layers, layer_labels=None, max_ticks=8):
    if n_layers <= max_ticks:
        ticks = list(range(n_layers))
    else:
        step = max(1, int(np.ceil(n_layers / max_ticks)))
        ticks = list(range(0, n_layers, step))

        if ticks[-1] != n_layers - 1:
            ticks.append(n_layers - 1)

    if layer_labels is not None and len(layer_labels) == n_layers:
        labels = [layer_labels[idx] for idx in ticks]
    else:
        labels = [f"L{idx}" for idx in ticks]
    return ticks, labels


def format_position_labels(positions):
    return [f"{int(round(position * 100))}%" for position in positions]


def build_metric_matrix(
    lens_out,
    metric_name,
    act_name,
    lens_mode,
    model_name,
    positions,
    sample_reducer,
):
    layer_vectors = []

    for position in positions:
        metric_tensor = lens_out[position][act_name][lens_mode][model_name][metric_name]
        mean_by_layer = sample_reducer(metric_tensor)

        if not th.is_tensor(mean_by_layer):
            mean_by_layer = th.as_tensor(mean_by_layer)

        if mean_by_layer.ndim != 1:
            raise ValueError(
                f"Reduced metric must have shape [L], got {tuple(mean_by_layer.shape)} "
                f"for metric={metric_name}, act_name={act_name}, model_name={model_name}"
            )

        layer_vectors.append(mean_by_layer.detach().cpu())

    return th.stack(layer_vectors, dim=1).numpy()


def build_metric_matrices(
    lens_out,
    metric_name,
    act_name,
    lens_mode,
    model_names,
    positions,
    sample_reducer,
):
    matrices = {}

    for model_name in model_names:
        matrices[model_name] = build_metric_matrix(
            lens_out=lens_out,
            metric_name=metric_name,
            act_name=act_name,
            lens_mode=lens_mode,
            model_name=model_name,
            positions=positions,
            sample_reducer=sample_reducer,
        )

    return matrices


def format_metric_markdown_tables(
    lens_out,
    metric_name,
    sample_reducer,
    lens_source_name=None,
    layer_labels=None,
    positions=None,
    act_names=None,
    lens_modes=None,
    model_names=None,
    decimals=4,
):
    positions = infer_positions(lens_out) if positions is None else positions
    act_names = infer_act_names(lens_out, positions) if act_names is None else act_names
    lens_modes = (
        infer_lens_modes(lens_out, positions, act_names)
        if lens_modes is None
        else lens_modes
    )

    rendered_tables = []
    position_labels = format_position_labels(positions)

    for act_name in act_names:
        for lens_mode in lens_modes:
            active_models = (
                infer_model_names(lens_out, positions, act_name, lens_mode)
                if model_names is None
                else model_names
            )

            panel_matrices = build_metric_matrices(
                lens_out=lens_out,
                metric_name=metric_name,
                act_name=act_name,
                lens_mode=lens_mode,
                model_names=active_models,
                positions=positions,
                sample_reducer=sample_reducer,
            )

            for model_name in active_models:
                matrix = panel_matrices[model_name]
                n_layers = matrix.shape[0]
                active_layer_labels = (
                    layer_labels if layer_labels is not None and len(layer_labels) == n_layers
                    else [f"L{idx}" for idx in range(n_layers)]
                )

                lines = [
                    (
                        f"### {METRIC_DISPLAY_NAMES.get(metric_name, metric_name)}"
                        f" | {ACT_DISPLAY_NAMES.get(act_name, act_name)}"
                        f" | {build_lens_display_label(lens_mode, lens_source_name)}"
                        f" | {model_name.upper()}"
                    ),
                    "",
                    "| Layer | " + " | ".join(position_labels) + " |",
                    "| --- | " + " | ".join(["---:"] * len(position_labels)) + " |",
                ]

                for layer_idx, layer_label in enumerate(active_layer_labels):
                    row_values = [
                        f"{float(matrix[layer_idx, pos_idx]):.{decimals}f}"
                        for pos_idx in range(len(position_labels))
                    ]
                    lines.append("| " + " | ".join([str(layer_label)] + row_values) + " |")

                rendered_tables.append("\n".join(lines))

    return rendered_tables


def plot_metric_panel(
    panel_matrices,
    positions,
    metric_name,
    act_name,
    lens_mode,
    output_path,
    lens_source_name=None,
    layer_labels=None,
):
    model_names = list(panel_matrices.keys())
    n_models = len(model_names)
    x_labels = format_position_labels(positions)

    arrays = [panel_matrices[model_name] for model_name in model_names]
    vmin = min(float(arr.min()) for arr in arrays)
    vmax = max(float(arr.max()) for arr in arrays)

    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6

    metric_label = METRIC_DISPLAY_NAMES.get(metric_name, metric_name.replace("_", " ").title())
    act_label = ACT_DISPLAY_NAMES.get(act_name, act_name)
    lens_label = build_lens_display_label(lens_mode, lens_source_name)

    fig_width = max(11, 3.8 * n_models + 1.0)
    fig_height = 5.8
    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(fig_width, fig_height),
        squeeze=False,
        sharey=True,
        gridspec_kw={"wspace": 0.16},
    )
    axes = axes[0]
    fig.subplots_adjust(left=0.08, right=0.9, top=0.83, bottom=0.17)

    last_im = None

    for idx, (ax, model_name) in enumerate(zip(axes, model_names)):
        matrix = panel_matrices[model_name]
        n_layers = matrix.shape[0]
        layer_ticks, active_layer_labels = build_layer_ticks(
            n_layers,
            layer_labels=layer_labels,
        )

        last_im = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(model_name.upper(), fontsize=12, pad=8, fontweight="semibold")
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=10)
        ax.set_yticks(layer_ticks)
        ax.tick_params(axis="y", labelsize=10)

        if idx == 0:
            ax.set_yticklabels(active_layer_labels, fontsize=10)
        else:
            ax.tick_params(labelleft=False)

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    axes[0].set_ylabel("Layer")
    fig.suptitle(
        f"Logit Lens {metric_label}\n{act_label} | {lens_label}",
        fontsize=15,
        fontweight="semibold",
        y=0.96,
    )
    fig.supxlabel("Completion Position", fontsize=11, y=0.08)

    cbar = fig.colorbar(last_im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label(metric_label, fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_plot_output_path(
    output_root,
    act_name,
    metric_name,
    lens_mode,
    lens_source_name=None,
):
    act_dir = ACT_FILE_STEMS.get(act_name, act_name)
    output_dir = os.path.join(
        output_root,
        act_dir,
        build_lens_variant_dirname(lens_mode, lens_source_name),
    )
    os.makedirs(output_dir, exist_ok=True)

    file_stem = ACT_FILE_STEMS.get(act_name, act_name)
    file_name = f"{file_stem}_{metric_name}_{lens_mode}_pos_mean.png"

    return os.path.join(output_dir, file_name)


def build_alignment_plot_output_path(
    output_root,
    act_name,
    metric_name,
    lens_mode,
    lens_source_name=None,
):
    act_dir = ACT_FILE_STEMS.get(act_name, act_name)
    output_dir = os.path.join(
        output_root,
        act_dir,
        build_lens_variant_dirname(lens_mode, lens_source_name),
    )
    os.makedirs(output_dir, exist_ok=True)

    file_stem = ACT_FILE_STEMS.get(act_name, act_name)
    file_name = f"{file_stem}_{metric_name}_{lens_mode}_generated_token_by_pos.png"

    return os.path.join(output_dir, file_name)


def plot_alignment_metric_small_multiples(
    panel_matrices,
    positions,
    metric_name,
    act_name,
    lens_mode,
    output_path,
    lens_source_name=None,
    layer_labels=None,
):
    model_names = list(panel_matrices.keys())
    metric_label = METRIC_DISPLAY_NAMES.get(metric_name, metric_name.replace("_", " ").title())
    act_label = ACT_DISPLAY_NAMES.get(act_name, act_name)
    lens_label = build_lens_display_label(lens_mode, lens_source_name)
    position_labels = format_position_labels(positions)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.5, 7.6),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig.subplots_adjust(left=0.08, right=0.84, top=0.84, bottom=0.12, wspace=0.16, hspace=0.24)
    flat_axes = axes.ravel()

    cmap = plt.get_cmap("tab10")
    color_map = {
        model_name: cmap(idx % cmap.N) for idx, model_name in enumerate(model_names)
    }

    max_layers = max(matrix.shape[0] for matrix in panel_matrices.values())
    x_ticks, x_tick_labels = build_layer_ticks(
        max_layers,
        layer_labels=layer_labels,
    )

    y_min = min(float(matrix.min()) for matrix in panel_matrices.values())
    y_max = max(float(matrix.max()) for matrix in panel_matrices.values())
    if np.isclose(y_min, y_max):
        y_max = y_min + 1e-6
    y_pad = max((y_max - y_min) * 0.08, 1e-6)

    legend_handles = []

    for ax_idx, (ax, position_label) in enumerate(zip(flat_axes, position_labels)):
        for model_idx, model_name in enumerate(model_names):
            matrix = panel_matrices[model_name]
            x_values = np.arange(matrix.shape[0])
            line = ax.plot(
                x_values,
                matrix[:, ax_idx],
                linewidth=2.0,
                color=color_map[model_name],
                label=model_name.upper(),
            )[0]

            if ax_idx == 0:
                legend_handles.append(line)

        ax.set_title(position_label, fontsize=12, pad=8, fontweight="semibold")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_tick_labels, fontsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(True, axis="y", alpha=0.24, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    for empty_ax in flat_axes[len(position_labels):]:
        empty_ax.axis("off")

    axes[0, 0].set_ylabel(metric_label, fontsize=11)
    axes[1, 0].set_ylabel(metric_label, fontsize=11)
    axes[1, 0].set_xlabel("Layer", fontsize=11)
    axes[1, 1].set_xlabel("Layer", fontsize=11)

    fig.suptitle(
        f"Generated-Token Alignment: {metric_label}\n{act_label} | {lens_label}",
        fontsize=15,
        fontweight="semibold",
        y=0.97,
    )
    fig.legend(
        handles=legend_handles,
        labels=[model_name.upper() for model_name in model_names],
        loc="center left",
        bbox_to_anchor=(0.86, 0.5),
        frameon=False,
        fontsize=10,
    )

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_sample_mean_metric_panels(
    lens_out,
    metric_names,
    sample_reducer,
    output_root,
    lens_source_name=None,
    layer_labels=None,
    positions=None,
    act_names=None,
    lens_modes=None,
    model_names=None,
):
    positions = infer_positions(lens_out) if positions is None else positions
    act_names = infer_act_names(lens_out, positions) if act_names is None else act_names
    lens_modes = (
        infer_lens_modes(lens_out, positions, act_names)
        if lens_modes is None
        else lens_modes
    )

    saved_paths = []

    for act_name in act_names:
        for lens_mode in lens_modes:
            active_models = (
                infer_model_names(lens_out, positions, act_name, lens_mode)
                if model_names is None
                else model_names
            )

            for metric_name in metric_names:
                panel_matrices = build_metric_matrices(
                    lens_out=lens_out,
                    metric_name=metric_name,
                    act_name=act_name,
                    lens_mode=lens_mode,
                    model_names=active_models,
                    positions=positions,
                    sample_reducer=sample_reducer,
                )

                output_path = build_plot_output_path(
                    output_root=output_root,
                    act_name=act_name,
                    metric_name=metric_name,
                    lens_mode=lens_mode,
                    lens_source_name=lens_source_name,
                )
                plot_metric_panel(
                    panel_matrices=panel_matrices,
                    positions=positions,
                    metric_name=metric_name,
                    act_name=act_name,
                    lens_mode=lens_mode,
                    output_path=output_path,
                    lens_source_name=lens_source_name,
                    layer_labels=layer_labels,
                )
                saved_paths.append(output_path)

    return saved_paths


def plot_alignment_metric_panels(
    lens_out,
    metric_names,
    sample_reducer,
    output_root,
    lens_source_name=None,
    layer_labels=None,
    positions=None,
    act_names=None,
    lens_modes=None,
    model_names=None,
):
    positions = infer_positions(lens_out) if positions is None else positions
    act_names = infer_act_names(lens_out, positions) if act_names is None else act_names
    lens_modes = (
        infer_lens_modes(lens_out, positions, act_names)
        if lens_modes is None
        else lens_modes
    )

    saved_paths = []

    for act_name in act_names:
        for lens_mode in lens_modes:
            active_models = (
                infer_model_names(lens_out, positions, act_name, lens_mode)
                if model_names is None
                else model_names
            )

            for metric_name in metric_names:
                panel_matrices = build_metric_matrices(
                    lens_out=lens_out,
                    metric_name=metric_name,
                    act_name=act_name,
                    lens_mode=lens_mode,
                    model_names=active_models,
                    positions=positions,
                    sample_reducer=sample_reducer,
                )

                output_path = build_alignment_plot_output_path(
                    output_root=output_root,
                    act_name=act_name,
                    metric_name=metric_name,
                    lens_mode=lens_mode,
                    lens_source_name=lens_source_name,
                )
                plot_alignment_metric_small_multiples(
                    panel_matrices=panel_matrices,
                    positions=positions,
                    metric_name=metric_name,
                    act_name=act_name,
                    lens_mode=lens_mode,
                    output_path=output_path,
                    lens_source_name=lens_source_name,
                    layer_labels=layer_labels,
                )
                saved_paths.append(output_path)

    return saved_paths

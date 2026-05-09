import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from models.model import get_tokenizer
from experiments.act_dataset_utils import RLVR_PATH


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
    "top2_prob": "Top-2 Probability",
    "prob_margin": "Probability Margin",
    "logit_margin": "Logit Margin",
    "target_logprob": "Target Log-Probability",
    "target_prob": "Target Probability",
    "surprisal": "Surprisal",
    "target_rank": "Target Rank",
}


_PLOT_TOKENIZER = None
_TOKENIZER_LOAD_FAILED = False


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


def reduce_metric_by_layer(metric_tensor, sample_reducer):
    reduced = sample_reducer(metric_tensor)

    if not th.is_tensor(reduced):
        reduced = th.as_tensor(reduced)

    if reduced.ndim != 1:
        raise ValueError(f"Reduced metric must have shape [L], got {tuple(reduced.shape)}")

    return reduced.detach().cpu()


def reduce_token_ids_by_mode(token_id_tensor):
    if not th.is_tensor(token_id_tensor):
        token_id_tensor = th.as_tensor(token_id_tensor)

    if token_id_tensor.ndim != 2:
        raise ValueError(
            f"Expected token-id tensor with shape [B, L], got {tuple(token_id_tensor.shape)}"
        )

    return th.mode(token_id_tensor.to(dtype=th.long), dim=0).values.detach().cpu()


def build_layer_labels(n_layers, layer_labels=None):
    if layer_labels is not None and len(layer_labels) == n_layers:
        return list(layer_labels)

    return [f"L{idx}" for idx in range(n_layers)]


def get_probability_margin_required_keys():
    return {
        "top1_prob",
        "top2_prob",
        "prob_margin",
        "top1_token_id",
        "top2_token_id",
    }


def has_probability_margin_table_inputs(
    lens_out,
    positions=None,
    act_names=None,
    lens_modes=None,
    model_names=None,
):
    positions = infer_positions(lens_out) if positions is None else positions
    if not positions:
        return False

    act_names = infer_act_names(lens_out, positions) if act_names is None else act_names
    lens_modes = (
        infer_lens_modes(lens_out, positions, act_names)
        if lens_modes is None
        else lens_modes
    )

    required_keys = get_probability_margin_required_keys()

    for act_name in act_names:
        for lens_mode in lens_modes:
            active_models = (
                infer_model_names(lens_out, positions, act_name, lens_mode)
                if model_names is None
                else model_names
            )
            for model_name in active_models:
                metric_ref = lens_out[positions[0]][act_name][lens_mode][model_name]
                if not required_keys.issubset(metric_ref.keys()):
                    return False

    return True


def get_plot_tokenizer(tokenizer_path=RLVR_PATH):
    global _PLOT_TOKENIZER, _TOKENIZER_LOAD_FAILED

    if _TOKENIZER_LOAD_FAILED:
        return None

    if _PLOT_TOKENIZER is None:
        try:
            _PLOT_TOKENIZER = get_tokenizer(tokenizer_path)
        except Exception:
            _TOKENIZER_LOAD_FAILED = True
            return None

    return _PLOT_TOKENIZER


def format_token_display(token_text, max_chars=18):
    safe_text = token_text
    if len(safe_text) <= max_chars:
        return safe_text

    return safe_text[: max_chars - 3] + "..."


def normalize_token_text(token_text):
    if token_text == "":
        return "<empty>"

    replacements = {
        " ": "<space>",
        "\n": "<newline>",
        "\t": "<tab>",
    }
    normalized = "".join(replacements.get(char, char) for char in token_text)

    if normalized.strip() == "":
        return normalized

    return normalized


def simplify_special_token(tokenizer, token_id):
    token_id = int(token_id)
    special_map = {}

    for attr_name, label in (
        ("eos_token_id", "<eos>"),
        ("bos_token_id", "<bos>"),
        ("pad_token_id", "<pad>"),
        ("unk_token_id", "<unk>"),
    ):
        attr_value = getattr(tokenizer, attr_name, None)
        if attr_value is not None:
            special_map[int(attr_value)] = label

    if token_id in special_map:
        return special_map[token_id]

    return None


def decode_token_id(tokenizer, token_id):
    if tokenizer is None:
        return f"<id:{int(token_id)}>"

    special_token = simplify_special_token(tokenizer, token_id)
    if special_token is not None:
        return special_token

    try:
        token_text = tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except Exception:
        return f"<id:{int(token_id)}>"

    if token_text is None:
        return f"<id:{int(token_id)}>"

    return format_token_display(normalize_token_text(token_text), max_chars=14)


def build_table_figure_height(n_rows, base_height=2.8, row_height=0.32, min_height=4.6):
    return max(min_height, base_height + n_rows * row_height)


def draw_table_on_axis(
    ax,
    col_labels,
    cell_text,
    title,
    font_size=8.0,
    y_scale=1.15,
    bbox=None,
    title_y=0.98,
    highlighted_cols=None,
):
    ax.axis("off")
    ax.text(
        0.5,
        title_y,
        title,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11.5,
        fontweight="semibold",
    )

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        bbox=bbox,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, y_scale)
    highlighted_cols = set() if highlighted_cols is None else set(highlighted_cols)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_linewidth(0.6)
        if row_idx == 0:
            cell.set_text_props(fontweight="semibold")
            cell.set_facecolor("#eef2f7")
        elif col_idx in highlighted_cols:
            cell.set_facecolor("#f7f9fc")

    return table




def format_metric_markdown_tables(
    lens_out,
    metric_name,
    sample_reducer,
    std_reducer=None,
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
            panel_std_matrices = None
            if std_reducer is not None:
                panel_std_matrices = build_metric_matrices(
                    lens_out=lens_out,
                    metric_name=metric_name,
                    act_name=act_name,
                    lens_mode=lens_mode,
                    model_names=active_models,
                    positions=positions,
                    sample_reducer=std_reducer,
                )

            for model_name in active_models:
                matrix = panel_matrices[model_name]
                std_matrix = None if panel_std_matrices is None else panel_std_matrices[model_name]
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
                    row_values = []
                    for pos_idx in range(len(position_labels)):
                        mean_value = float(matrix[layer_idx, pos_idx])
                        if std_matrix is None:
                            row_values.append(f"{mean_value:.{decimals}f}")
                            continue

                        std_value = float(std_matrix[layer_idx, pos_idx])
                        row_values.append(
                            f"{mean_value:.{decimals}f} ± {std_value:.{decimals}f}"
                        )
                    lines.append("| " + " | ".join([str(layer_label)] + row_values) + " |")

                rendered_tables.append("\n".join(lines))

    return rendered_tables


def plot_metric_table_panel(
    panel_matrices,
    positions,
    metric_name,
    act_name,
    lens_mode,
    output_path,
    lens_source_name=None,
    layer_labels=None,
    decimals=4,
):
    model_names = list(panel_matrices.keys())
    position_labels = format_position_labels(positions)

    metric_label = METRIC_DISPLAY_NAMES.get(metric_name, metric_name.replace("_", " ").title())
    act_label = ACT_DISPLAY_NAMES.get(act_name, act_name)
    lens_label = build_lens_display_label(lens_mode, lens_source_name)

    max_layers = max(matrix.shape[0] for matrix in panel_matrices.values())
    fig_width = max(9.6, 3.5 * len(model_names))
    fig_height = max(2.8, 1.45 + 0.42 * max_layers)
    fig, axes = plt.subplots(
        1,
        len(model_names),
        figsize=(fig_width, fig_height),
        squeeze=False,
    )
    axes = axes[0]
    fig.subplots_adjust(left=0.02, right=0.985, top=0.74, bottom=0.08, wspace=0.16)

    for ax, model_name in zip(axes, model_names):
        matrix = panel_matrices[model_name]
        n_layers = matrix.shape[0]
        active_layer_labels = build_layer_labels(n_layers, layer_labels=layer_labels)
        cell_text = [
            [str(active_layer_labels[layer_idx])]
            + [
                f"{float(matrix[layer_idx, pos_idx]):.{decimals}f}"
                for pos_idx in range(len(position_labels))
            ]
            for layer_idx in range(n_layers)
        ]
        font_size = 8.2 if n_layers <= 8 else 7.2
        y_scale = 1.06 if n_layers <= 8 else 1.0
        draw_table_on_axis(
            ax=ax,
            col_labels=["Layer"] + position_labels,
            cell_text=cell_text,
            title=model_name.upper(),
            font_size=font_size,
            y_scale=y_scale,
            bbox=[0.0, 0.02, 1.0, 0.72],
            title_y=0.82,
            highlighted_cols={0},
        )

    fig.suptitle(
        f"Logit Lens {metric_label} Raw Values\n{act_label} | {lens_label}",
        fontsize=13,
        fontweight="semibold",
        y=0.96,
    )
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


def build_model_plot_output_path(
    output_root,
    act_name,
    metric_name,
    lens_mode,
    model_name,
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
    file_name = f"{file_stem}_{metric_name}_{lens_mode}_{model_name}_raw_table.png"

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


def build_probability_margin_table_blocks(
    lens_out,
    act_name,
    lens_mode,
    model_name,
    positions,
    sample_reducer,
    tokenizer=None,
    layer_labels=None,
    decimals=4,
):
    position_labels = format_position_labels(positions)
    table_blocks = []
    active_layer_labels = None

    for position_idx, position in enumerate(positions):
        metric_ref = lens_out[position][act_name][lens_mode][model_name]
        top1_prob = reduce_metric_by_layer(metric_ref["top1_prob"], sample_reducer)
        top2_prob = reduce_metric_by_layer(metric_ref["top2_prob"], sample_reducer)
        prob_margin = reduce_metric_by_layer(metric_ref["prob_margin"], sample_reducer)
        top1_ids = reduce_token_ids_by_mode(metric_ref["top1_token_id"])
        top2_ids = reduce_token_ids_by_mode(metric_ref["top2_token_id"])

        n_layers = top1_prob.shape[0]
        active_layer_labels = build_layer_labels(n_layers, layer_labels=layer_labels)
        table_blocks.append(
            {
                "position_label": position_labels[position_idx],
                "top1_prob": top1_prob,
                "top2_prob": top2_prob,
                "prob_margin": prob_margin,
                "top1_ids": top1_ids,
                "top2_ids": top2_ids,
            }
        )

    if active_layer_labels is None:
        return []

    layer_tables = []
    for layer_idx, layer_label in enumerate(active_layer_labels):
        rows = []
        for block in table_blocks:
            rows.append(
                [
                    block["position_label"],
                    f"{float(block['top1_prob'][layer_idx]):.{decimals}f}",
                    f"{float(block['top2_prob'][layer_idx]):.{decimals}f}",
                    f"{float(block['prob_margin'][layer_idx]):.{decimals}f}",
                ]
            )
        layer_tables.append((str(layer_label), rows))

    return layer_tables


def plot_probability_margin_table(
    lens_out,
    positions,
    act_name,
    lens_mode,
    model_name,
    output_path,
    sample_reducer,
    lens_source_name=None,
    layer_labels=None,
):
    metric_label = METRIC_DISPLAY_NAMES["prob_margin"]
    act_label = ACT_DISPLAY_NAMES.get(act_name, act_name)
    lens_label = build_lens_display_label(lens_mode, lens_source_name)
    tokenizer = get_plot_tokenizer()

    layer_tables = build_probability_margin_table_blocks(
        lens_out=lens_out,
        act_name=act_name,
        lens_mode=lens_mode,
        model_name=model_name,
        positions=positions,
        sample_reducer=sample_reducer,
        tokenizer=tokenizer,
        layer_labels=layer_labels,
    )

    col_labels = [
        "Pos",
        "P(top-1)",
        "P(top-2)",
        "Margin",
    ]
    fig_width = 13.8
    fig_height = max(5.2, 1.65 * len(layer_tables) + 1.1)

    fig, axes = plt.subplots(len(layer_tables), 1, figsize=(fig_width, fig_height), squeeze=False)
    axes = axes[:, 0]
    fig.subplots_adjust(left=0.03, right=0.99, top=0.82, bottom=0.05, hspace=0.28)

    for ax, (layer_label, rows) in zip(axes, layer_tables):
        draw_table_on_axis(
            ax=ax,
            col_labels=col_labels,
            cell_text=rows,
            title=f"Layer {layer_label}",
            font_size=7.8,
            y_scale=1.0,
            bbox=[0.0, 0.02, 1.0, 0.76],
            title_y=0.86,
            highlighted_cols={1, 2},
        )

    fig.suptitle(
        (
            f"Logit Lens {metric_label} Raw Values | {model_name.upper()}\n"
            f"{act_label} | {lens_label}"
        ),
        fontsize=13,
        fontweight="semibold",
        y=0.965,
    )
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


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
                if metric_name == "prob_margin" and has_probability_margin_table_inputs(
                    lens_out=lens_out,
                    positions=positions,
                    act_names=[act_name],
                    lens_modes=[lens_mode],
                    model_names=active_models,
                ):
                    for model_name in active_models:
                        output_path = build_model_plot_output_path(
                            output_root=output_root,
                            act_name=act_name,
                            metric_name=metric_name,
                            lens_mode=lens_mode,
                            model_name=model_name,
                            lens_source_name=lens_source_name,
                        )
                        plot_probability_margin_table(
                            lens_out=lens_out,
                            positions=positions,
                            act_name=act_name,
                            lens_mode=lens_mode,
                            model_name=model_name,
                            output_path=output_path,
                            sample_reducer=sample_reducer,
                            lens_source_name=lens_source_name,
                            layer_labels=layer_labels,
                        )
                        saved_paths.append(output_path)
                    continue

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
                plot_metric_table_panel(
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

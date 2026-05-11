import torch as th

RESIDUAL_ACT_NAMES = ("resid_pre", "attn_resid", "mlp_resid")


def get_target_logprob(lens_out, position, act_name, lens_mode, model_name):
    return lens_out[position][act_name][lens_mode][model_name]["target_logprob"]


def get_pairwise_metric(lens_out, position, act_name, lens_mode, comparison_name, metric_name):
    return lens_out[position][act_name][lens_mode][comparison_name][metric_name]


def component_logprob_gain(lens_out, position, lens_mode, model_name):
    resid_pre = get_target_logprob(
        lens_out=lens_out,
        position=position,
        act_name="resid_pre",
        lens_mode=lens_mode,
        model_name=model_name,
    )

    attn_resid = get_target_logprob(
        lens_out=lens_out,
        position=position,
        act_name="attn_resid",
        lens_mode=lens_mode,
        model_name=model_name,
    )

    mlp_resid = get_target_logprob(
        lens_out=lens_out,
        position=position,
        act_name="mlp_resid",
        lens_mode=lens_mode,
        model_name=model_name,
    )

    return {
        "delta_attn_logprob": attn_resid - resid_pre,
        "delta_mlp_logprob": mlp_resid - attn_resid,
    }


def delta_pairwise_DKL(lens_out, position, lens_mode, comparison_name):
    attn_dkl = get_pairwise_metric(
        lens_out=lens_out,
        position=position,
        act_name="attn_resid",
        lens_mode=lens_mode,
        comparison_name=comparison_name,
        metric_name="kl_divergence",
    )
    mlp_dkl = get_pairwise_metric(
        lens_out=lens_out,
        position=position,
        act_name="mlp_resid",
        lens_mode=lens_mode,
        comparison_name=comparison_name,
        metric_name="kl_divergence",
    )

    return mlp_dkl - attn_dkl


def delta_jaccard(lens_out, position, lens_mode, comparison_name):
    attn_jaccard = get_pairwise_metric(
        lens_out=lens_out,
        position=position,
        act_name="attn_resid",
        lens_mode=lens_mode,
        comparison_name=comparison_name,
        metric_name="topk_jaccard",
    )
    mlp_jaccard = get_pairwise_metric(
        lens_out=lens_out,
        position=position,
        act_name="mlp_resid",
        lens_mode=lens_mode,
        comparison_name=comparison_name,
        metric_name="topk_jaccard",
    )

    return mlp_jaccard - attn_jaccard


def as_sample_layer_tensor(metric_tensor):
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


def median_iqr_by_layer(metric_tensor):
    metric_tensor = as_sample_layer_tensor(metric_tensor)
    median = metric_tensor.quantile(0.5, dim=0)
    iqr = metric_tensor.quantile(0.75, dim=0) - metric_tensor.quantile(0.25, dim=0)
    return median, iqr


def build_target_logprob_residual_rows(
    lens_out,
    positions,
    lens_modes,
    model_names,
    layer_labels=None,
):
    rows = []

    for position in positions:
        for lens_mode in lens_modes:
            for model_name in model_names:
                try:
                    summaries = {
                        act_name: median_iqr_by_layer(
                            get_target_logprob(
                                lens_out=lens_out,
                                position=position,
                                act_name=act_name,
                                lens_mode=lens_mode,
                                model_name=model_name,
                            )
                        )
                        for act_name in RESIDUAL_ACT_NAMES
                    }
                except KeyError:
                    continue

                n_layers = len(summaries["resid_pre"][0])
                for layer_idx in range(n_layers):
                    layer_label = (
                        layer_labels[layer_idx]
                        if layer_labels is not None and len(layer_labels) == n_layers
                        else str(layer_idx)
                    )
                    row = {
                        "position": position,
                        "lens_mode": lens_mode,
                        "model": model_name,
                        "layer": layer_label,
                    }

                    for act_name, (median, iqr) in summaries.items():
                        row[f"{act_name}_median"] = float(median[layer_idx])
                        row[f"{act_name}_iqr"] = float(iqr[layer_idx])

                    rows.append(row)

    return rows


def format_target_logprob_residual_table(
    lens_out,
    positions,
    lens_modes,
    model_names,
    layer_labels=None,
    decimals=4,
):
    rows = build_target_logprob_residual_rows(
        lens_out=lens_out,
        positions=positions,
        lens_modes=lens_modes,
        model_names=model_names,
        layer_labels=layer_labels,
    )

    headers = (
        "position",
        "lens_mode",
        "model",
        "layer",
        "target_logprob(resid_pre) median",
        "IQR",
        "target_logprob(attn_resid) median",
        "IQR",
        "target_logprob(mlp_resid) median",
        "IQR",
    )
    lines = ["\t".join(headers)]

    for row in rows:
        lines.append(
            "\t".join(
                (
                    f"{float(row['position']):.2f}",
                    str(row["lens_mode"]),
                    str(row["model"]),
                    str(row["layer"]),
                    f"{row['resid_pre_median']:.{decimals}f}",
                    f"{row['resid_pre_iqr']:.{decimals}f}",
                    f"{row['attn_resid_median']:.{decimals}f}",
                    f"{row['attn_resid_iqr']:.{decimals}f}",
                    f"{row['mlp_resid_median']:.{decimals}f}",
                    f"{row['mlp_resid_iqr']:.{decimals}f}",
                )
            )
        )

    return "\n".join(lines)

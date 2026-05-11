import os
import sys
from dataclasses import dataclass
from itertools import combinations

import torch as th

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from experiments.act_dataset_utils import (
    RLVR_PATH,
    SFT_PATH,
    format_layer_group_label,
    get_activation_dataset,
    load_sample_batch,
)
from experiments.Logit_Lens.get_lens import get_lens
from experiments.Logit_Lens.lens_plot import (
    build_pairwise_plot_output_path,
    infer_comparison_names,
    infer_model_names,
    infer_positions,
    plot_alignment_metric_panels,
    plot_component_delta_tables,
    plot_component_metrics,
    plot_pairwise_dkl_delta,
    plot_pairwise_jaccard_delta,
)
from experiments.Logit_Lens.lens_metrics_derived import format_target_logprob_residual_table
from experiments.Logit_Lens.lens_utils import lens_view


LENS_OUT_METADATA_KEY = "__metadata__"
CACHE_SCHEMA_VERSION = "minimal_logit_lens_v2_symmetric_dkl"

SINGLE_MODEL_METRICS = (
    "target_logprob",
    "target_rank",
)

PAIRWISE_METRICS = (
    "kl_divergence",
    "topk_jaccard",
)


@dataclass(frozen=True)
class RunConfig:
    positions: tuple = (0.1, 0.5, 0.9, 0.95)
    act_modules: tuple = ("resid_pre", "attn_resid", "mlp_resid")
    report_act_modules: tuple = ("attn_resid", "mlp_resid")
    lens_modes: tuple = ("native",)
    shared_lens_source: str = "rlvr"
    layers: object = None
    batch_size: int = 10
    max_sample: int = 100
    sample_selection_seed: int = 42


@dataclass
class ActivationBatch:
    activation_out: dict
    model_names: list
    layer_labels: list
    sample_ids: list


def get_lens_output_dir():
    return os.path.join(project_root, "experiments", "Logit_Lens", "outputs")


def get_logit_lens_img_dir():
    return os.path.join(project_root, "experiments", "Logit_Lens", "logit_lens_img")


def resolve_lens_output_path(lens_modes, shared_lens_source):
    lens_mode_tag = "_".join(lens_modes)
    output_name = f"lens_out_{lens_mode_tag}_shared-{shared_lens_source}.pt"
    return os.path.join(get_lens_output_dir(), output_name)


def resolve_token_cache_path(h5_path):
    activation_dir = os.path.dirname(h5_path)
    dataset_name = os.path.splitext(os.path.basename(h5_path))[0]

    if not dataset_name.endswith("_acts"):
        raise ValueError(f"Unexpected activation dataset name: {dataset_name}")

    token_prefix = dataset_name.removesuffix("_acts").replace("_max_", "_max")
    token_cache_name = f"{token_prefix}_tok.pt"
    token_cache_path = os.path.join(activation_dir, "tokens_cache", token_cache_name)

    if not os.path.exists(token_cache_path):
        raise FileNotFoundError(f"Token cache not found: {token_cache_path}")

    return token_cache_path


def normalize_metadata_value(value):
    if th.is_tensor(value):
        return value.detach().cpu().tolist()

    if isinstance(value, tuple):
        return [normalize_metadata_value(item) for item in value]

    if isinstance(value, list):
        return [normalize_metadata_value(item) for item in value]

    return value


def abs_path_or_none(path):
    if path is None:
        return None

    return os.path.abspath(path)


def build_cache_metadata(
    config,
    h5_path,
    token_cache_path,
    model_names=None,
    layer_labels=None,
    sample_ids=None,
):
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "h5_path": abs_path_or_none(h5_path),
        "token_cache_path": abs_path_or_none(token_cache_path),
        "positions": normalize_metadata_value(config.positions),
        "act_modules": normalize_metadata_value(config.act_modules),
        "lens_modes": normalize_metadata_value(config.lens_modes),
        "shared_lens_source": config.shared_lens_source,
        "layers": normalize_metadata_value(config.layers),
        "batch_size": config.batch_size,
        "max_sample": config.max_sample,
        "sample_selection_seed": config.sample_selection_seed,
    }

    if model_names is not None:
        metadata["model_names"] = normalize_metadata_value(model_names)

    if layer_labels is not None:
        metadata["layer_labels"] = normalize_metadata_value(layer_labels)

    if sample_ids is not None:
        metadata["sample_ids"] = normalize_metadata_value(sample_ids)

    return metadata


def attach_metadata(lens_out, metadata):
    lens_out = dict(lens_out)
    lens_out[LENS_OUT_METADATA_KEY] = metadata
    return lens_out


def metadata_matches(cached_metadata, expected_metadata):
    if not isinstance(cached_metadata, dict):
        return False

    for key, expected_value in expected_metadata.items():
        if cached_metadata.get(key) != expected_value:
            print(f"Cache mismatch on metadata field: {key}")
            print(f"cached   = {cached_metadata.get(key)}")
            print(f"expected = {expected_value}")
            return False

    return True


def has_required_metrics(lens_out):
    for position in infer_positions(lens_out):
        for act_name, act_ref in lens_out[position].items():
            for lens_mode, lens_mode_ref in act_ref.items():
                for series_name, metric_ref in lens_mode_ref.items():
                    if "_vs_" in series_name:
                        missing = set(PAIRWISE_METRICS).difference(metric_ref.keys())
                    else:
                        missing = set(SINGLE_MODEL_METRICS).difference(metric_ref.keys())

                    if missing:
                        print(
                            "Cache missing required metrics "
                            f"position={position}, act_name={act_name}, "
                            f"lens_mode={lens_mode}, series={series_name}, "
                            f"missing={sorted(missing)}"
                        )
                        return False

    return True


def validate_lens_out_cache(lens_out, expected_metadata):
    cached_metadata = lens_out.get(LENS_OUT_METADATA_KEY)

    if not metadata_matches(cached_metadata, expected_metadata):
        return False

    if not has_required_metrics(lens_out):
        return False

    return True


def try_load_lens_out(output_path, expected_metadata):
    if not os.path.exists(output_path):
        return None

    print("=== LOADING EXISTING LENS OUT ===")
    lens_out = th.load(output_path, map_location="cpu")

    if validate_lens_out_cache(lens_out, expected_metadata):
        return lens_out

    print("=== EXISTING LENS OUT CACHE INVALID: RECOMPUTING ===")
    return None


def save_lens_out(lens_out, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    th.save(lens_out, output_path)
    return output_path


def build_model_path_map():
    return {
        "base": None,
        "sftt": SFT_PATH,
        "rlvr": RLVR_PATH,
    }


def build_bank_lens(model_names, device, shared_lens_source):
    model_paths = build_model_path_map()
    native_lenses = {}

    for model_name in model_names:
        if model_name not in model_paths:
            raise ValueError(f"Unknown model name: {model_name}")

        native_lenses[model_name] = get_lens(model_paths[model_name], device)

    if shared_lens_source not in native_lenses:
        raise ValueError(f"Unknown shared lens source: {shared_lens_source}")

    return {
        "native": native_lenses,
        "shared": native_lenses[shared_lens_source],
    }


def create_model_comparison(model_names):
    return list(combinations(model_names, 2))


def load_activation_batch(config, h5_path):
    activation_out, layer_ids = load_sample_batch(
        batch_size=config.batch_size,
        model_names=None,
        act_modules=list(config.act_modules),
        layers=config.layers,
        h5_path=h5_path,
        max_sample=config.max_sample,
        sample_seed=config.sample_selection_seed,
    )

    model_names = list(activation_out.keys())

    if not model_names:
        raise ValueError("No models found in activation_out.")

    layer_labels = [format_layer_group_label(layer_id) for layer_id in layer_ids]

    base_model_name = model_names[0]
    sample_ids = sorted(
        activation_out[base_model_name][config.act_modules[0]].keys()
    )

    return ActivationBatch(
        activation_out=activation_out,
        model_names=model_names,
        layer_labels=layer_labels,
        sample_ids=sample_ids,
    )


def get_or_compute_lens_out(config):
    device = "cuda" if th.cuda.is_available() else "cpu"

    act_dataset = get_activation_dataset()
    h5_path = act_dataset["h5_path"]
    token_cache_path = resolve_token_cache_path(h5_path)

    output_path = resolve_lens_output_path(
        lens_modes=config.lens_modes,
        shared_lens_source=config.shared_lens_source,
    )

    minimal_expected_metadata = build_cache_metadata(
        config=config,
        h5_path=h5_path,
        token_cache_path=token_cache_path,
    )

    cached_lens_out = try_load_lens_out(
        output_path=output_path,
        expected_metadata=minimal_expected_metadata,
    )

    if cached_lens_out is not None:
        metadata = cached_lens_out.get(LENS_OUT_METADATA_KEY, {})
        return (
            cached_lens_out,
            output_path,
            metadata.get("layer_labels"),
        )

    print("=== LOADING ACTIVATION BATCH ===")
    batch = load_activation_batch(config=config, h5_path=h5_path)

    full_metadata = build_cache_metadata(
        config=config,
        h5_path=h5_path,
        token_cache_path=token_cache_path,
        model_names=batch.model_names,
        layer_labels=batch.layer_labels,
        sample_ids=batch.sample_ids,
    )

    print("=== COMPUTING LENS OUT ===")
    lens_bank = build_bank_lens(
        model_names=batch.model_names,
        device=device,
        shared_lens_source=config.shared_lens_source,
    )

    lens_out = lens_view(
        views=list(config.positions),
        act_names=list(config.act_modules),
        lens_modes=list(config.lens_modes),
        model_names=batch.model_names,
        activation_out=batch.activation_out,
        lens_bank=lens_bank,
        model_comparison=create_model_comparison(batch.model_names),
        device=device,
        token_cache_path=token_cache_path,
    )

    lens_out = attach_metadata(lens_out, full_metadata)
    save_lens_out(lens_out, output_path)

    return lens_out, output_path, batch.layer_labels


def median_metric_over_samples(metric_tensor):
    if not th.is_tensor(metric_tensor):
        metric_tensor = th.as_tensor(metric_tensor)

    while metric_tensor.ndim > 2 and metric_tensor.shape[-1] == 1:
        metric_tensor = metric_tensor.squeeze(-1)

    if metric_tensor.ndim != 2:
        raise ValueError(
            f"Expected metric tensor with shape [B, L], got {tuple(metric_tensor.shape)}"
        )

    if not th.is_floating_point(metric_tensor):
        metric_tensor = metric_tensor.to(dtype=th.float32)

    return metric_tensor.quantile(0.5, dim=0)


def build_single_model_plots(lens_out, config, layer_labels=None):
    # Single-model primitive metrics are kept in lens_out for derived metrics
    # and cache validation, but are not part of the Logit Lens report plots.
    return []


def build_pairwise_plots(lens_out, config, layer_labels=None):
    comparison_names = infer_comparison_names(lens_out)

    if not comparison_names:
        return []

    output_root = get_logit_lens_img_dir()
    os.makedirs(output_root, exist_ok=True)

    saved_paths = plot_alignment_metric_panels(
        lens_out=lens_out,
        metric_names=list(PAIRWISE_METRICS),
        output_root=output_root,
        lens_source_name=config.shared_lens_source,
        layer_labels=layer_labels,
        positions=list(config.positions),
        act_names=list(config.report_act_modules),
        model_names=comparison_names,
        output_path_builder=build_pairwise_plot_output_path,
        title_prefix="Pairwise Model Comparison",
        sample_reducer=median_metric_over_samples,
    )
    saved_paths.extend(
        plot_pairwise_dkl_delta(
            lens_out=lens_out,
            output_root=output_root,
            layer_labels=layer_labels,
            positions=list(config.positions),
            lens_modes=list(config.lens_modes),
            sample_reducer=median_metric_over_samples,
        )
    )
    saved_paths.extend(
        plot_pairwise_jaccard_delta(
            lens_out=lens_out,
            output_root=output_root,
            layer_labels=layer_labels,
            positions=list(config.positions),
            lens_modes=list(config.lens_modes),
            sample_reducer=median_metric_over_samples,
        )
    )

    return saved_paths


def build_component_plots(lens_out, config, layer_labels=None):
    output_root = get_logit_lens_img_dir()
    os.makedirs(output_root, exist_ok=True)

    saved_paths = plot_component_metrics(
        lens_out=lens_out,
        output_root=output_root,
        layer_labels=layer_labels,
        positions=list(config.positions),
        lens_modes=list(config.lens_modes),
        sample_reducer=median_metric_over_samples,
    )
    saved_paths.extend(
        plot_component_delta_tables(
            lens_out=lens_out,
            output_root=output_root,
            layer_labels=layer_labels,
            positions=list(config.positions),
            lens_modes=list(config.lens_modes),
            sample_reducer=median_metric_over_samples,
        )
    )

    return saved_paths


def print_target_logprob_residual_table(lens_out, config, layer_labels=None):
    print("=== TARGET LOGPROB RESIDUAL CHECK (median/IQR over samples) ===")
    print(
        format_target_logprob_residual_table(
            lens_out=lens_out,
            positions=list(config.positions),
            lens_modes=list(config.lens_modes),
            model_names=infer_model_names(lens_out),
            layer_labels=layer_labels,
        )
    )


def run_reports(lens_out, config, layer_labels=None):
    print("=== BUILDING LOGIT LENS PLOTS ===")
    pairwise_plot_paths = build_pairwise_plots(
        lens_out=lens_out,
        config=config,
        layer_labels=layer_labels,
    )
    component_plot_paths = build_component_plots(
        lens_out=lens_out,
        config=config,
        layer_labels=layer_labels,
    )

    print("Saved pairwise comparison plots:")
    for path in pairwise_plot_paths:
        print(path)

    print("Saved component delta plots:")
    for path in component_plot_paths:
        print(path)

    print_target_logprob_residual_table(
        lens_out=lens_out,
        config=config,
        layer_labels=layer_labels,
    )


def main():
    config = RunConfig()

    lens_out, output_path, layer_labels = get_or_compute_lens_out(config)

    print("Available lens modes:", list(config.lens_modes))
    print("Shared lens source:", config.shared_lens_source)
    print("Computed positions:", infer_positions(lens_out))
    print("Saved lens_out to:", output_path)
    print("Resolved layer labels:", layer_labels)

    run_reports(
        lens_out=lens_out,
        config=config,
        layer_labels=layer_labels,
    )


if __name__ == "__main__":
    main()

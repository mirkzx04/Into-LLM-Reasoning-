import os
import sys
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
    resolve_saved_layer_labels,
)
from experiments.Logit_Lens.get_lens import get_lens
from experiments.Logit_Lens.lens_plot import (
    format_metric_markdown_tables,
    has_probability_margin_table_inputs,
    infer_positions,
    plot_alignment_metric_panels,
    plot_sample_mean_metric_panels,
)
from experiments.Logit_Lens.lens_utils import lens_view


LENS_OUT_METADATA_KEY = "__metadata__"
MISSING_METADATA_FIELD = "<missing>"


def create_model_comparison(model_names):
    # Build all model pairs used by pairwise metrics.
    model_pairs = list(combinations(model_names, 2))
    return model_pairs


def build_model_path_map():
    # "base" uses the default HF model.
    # The other entries point to local checkpoints.
    model_paths = {
        "base": None,
        "sftt": SFT_PATH,
        "rlvr": RLVR_PATH,
    }
    return model_paths


def build_bank_lens(
    model_names,
    device,
    shared_lens_source,
):
    # bank_lens["native"][model_name]:
    # each model uses its own lens.
    #
    # bank_lens["shared"]:
    # all models reuse the lens chosen by shared_lens_source.
    model_paths = build_model_path_map()
    native_lenses = {}

    for model_name in model_names:
        if model_name not in model_paths:
            raise ValueError(f"Unknown model name: {model_name}")

        model_path = model_paths[model_name]
        lens = get_lens(model_path, device)

        native_lenses[model_name] = lens

    if shared_lens_source not in native_lenses:
        raise ValueError(f"Unknown shared lens source: {shared_lens_source}")

    bank_lens = {
        "native": native_lenses,
        "shared": native_lenses[shared_lens_source],
    }
    return bank_lens


def resolve_token_cache_path(h5_path):
    # The token cache is stored next to the activation dataset.
    activation_dir = os.path.dirname(h5_path)
    dataset_name = os.path.splitext(os.path.basename(h5_path))[0]

    if not dataset_name.endswith("_acts"):
        raise ValueError(f"Unexpected activation dataset name: {dataset_name}")

    token_prefix = dataset_name.removesuffix("_acts").replace("_max_", "_max")
    token_cache_name = f"{token_prefix}_tok.pt"
    token_cache_path = os.path.join(
        activation_dir,
        "tokens_cache",
        token_cache_name,
    )

    if not os.path.exists(token_cache_path):
        raise FileNotFoundError(f"Token cache not found: {token_cache_path}")

    return token_cache_path


def save_lens_out(
    lens_out,
    lens_modes,
    shared_lens_source,
):
    # Save lens_out to a local .pt file.
    output_dir = get_lens_output_dir()
    os.makedirs(output_dir, exist_ok=True)
    output_path = resolve_lens_output_path(lens_modes, shared_lens_source)

    th.save(lens_out, output_path)
    return output_path


def normalize_metadata_value(value):
    if th.is_tensor(value):
        return value.detach().cpu().tolist()

    if isinstance(value, tuple):
        return [normalize_metadata_value(item) for item in value]

    if isinstance(value, list):
        return [normalize_metadata_value(item) for item in value]

    return value


def build_file_metadata(file_path):
    if file_path is None:
        return None, None, None

    normalized_path = os.path.abspath(file_path)

    if not os.path.exists(normalized_path):
        return normalized_path, None, None

    file_stats = os.stat(normalized_path)
    return normalized_path, file_stats.st_mtime, file_stats.st_size


def build_lens_out_metadata(
    h5_path=None,
    token_cache_path=None,
    positions=None,
    act_modules=None,
    lens_modes=None,
    shared_lens_source=None,
    max_sample=None,
    batch_size=None,
    sample_selection_seed=None,
    layers=None,
    layer_labels=None,
    model_names=None,
    sample_ids=None,
):
    h5_path, h5_mtime, h5_size = build_file_metadata(h5_path)
    token_cache_path, token_cache_mtime, _token_cache_size = build_file_metadata(token_cache_path)

    metadata = {
        "h5_path": h5_path,
        "h5_mtime": h5_mtime,
        "h5_size": h5_size,
        "token_cache_path": token_cache_path,
        "token_cache_mtime": token_cache_mtime,
        "positions": normalize_metadata_value(positions),
        "act_modules": normalize_metadata_value(act_modules),
        "lens_modes": normalize_metadata_value(lens_modes),
        "shared_lens_source": shared_lens_source,
        "max_sample": max_sample,
        "batch_size": batch_size,
        "sample_selection_seed": sample_selection_seed,
        "layers": normalize_metadata_value(layers),
        "layer_labels": normalize_metadata_value(layer_labels),
        "model_names": normalize_metadata_value(model_names),
        "sample_ids": normalize_metadata_value(sample_ids),
    }

    return metadata


def attach_lens_out_metadata(
    lens_out,
    h5_path=None,
    token_cache_path=None,
    positions=None,
    act_modules=None,
    lens_modes=None,
    max_sample=None,
    batch_size=None,
    sample_selection_seed=None,
    layers=None,
    layer_labels=None,
    model_names=None,
    sample_ids=None,
    shared_lens_source=None,
):
    lens_out_with_metadata = dict(lens_out)
    metadata = build_lens_out_metadata(
        h5_path=h5_path,
        token_cache_path=token_cache_path,
        positions=positions,
        act_modules=act_modules,
        lens_modes=lens_modes,
        max_sample=max_sample,
        batch_size=batch_size,
        sample_selection_seed=sample_selection_seed,
        layers=layers,
        layer_labels=layer_labels,
        model_names=model_names,
        sample_ids=sample_ids,
        shared_lens_source=shared_lens_source,
    )

    lens_out_with_metadata[LENS_OUT_METADATA_KEY] = metadata

    return lens_out_with_metadata


def validate_lens_out_cache(lens_out, expected_metadata):
    cached_metadata = lens_out.get(LENS_OUT_METADATA_KEY)

    if not isinstance(cached_metadata, dict):
        print("Cache invalid: missing '__metadata__'.")
        return False

    is_valid = True

    for field_name, expected_value in expected_metadata.items():
        cached_value = cached_metadata.get(field_name, MISSING_METADATA_FIELD)

        if cached_value != expected_value:
            print(f"Cache mismatch on field '{field_name}':")
            print(f"cached = {cached_value}")
            print(f"expected = {expected_value}")
            is_valid = False

    return is_valid


def resolve_default_activation_h5_path():
    activation_dir = os.path.join(project_root, "activation_dataset")

    if not os.path.isdir(activation_dir):
        return None

    h5_paths = sorted(
        os.path.join(activation_dir, entry.name)
        for entry in os.scandir(activation_dir)
        if entry.is_file() and entry.name.endswith(".h5")
    )

    if not h5_paths:
        return None

    if len(h5_paths) == 1:
        return h5_paths[0]

    return max(h5_paths, key=os.path.getmtime)


def resolve_layer_labels_for_lens_out(lens_out, h5_path=None, layers=None):
    metadata = lens_out.get(LENS_OUT_METADATA_KEY, {})
    saved_layer_labels = metadata.get("layer_labels")
    if saved_layer_labels:
        return saved_layer_labels

    candidate_h5_path = h5_path or metadata.get("h5_path") or resolve_default_activation_h5_path()
    if candidate_h5_path is None or not os.path.exists(candidate_h5_path):
        return None

    return resolve_saved_layer_labels(
        h5_path=candidate_h5_path,
        layers=layers,
    )


def get_lens_output_dir():
    return os.path.join(project_root, "experiments", "Logit_Lens", "outputs")


def get_logit_lens_img_dir():
    return os.path.join(project_root, "experiments", "Logit_Lens", "logit_lens_img")


def resolve_lens_output_path(lens_modes, shared_lens_source):
    lens_mode_tag = "_".join(lens_modes)
    output_name = f"lens_out_{lens_mode_tag}_shared-{shared_lens_source}.pt"
    return os.path.join(get_lens_output_dir(), output_name)


def has_populated_lens_outputs():
    output_dir = get_lens_output_dir()

    if not os.path.isdir(output_dir):
        return False

    return any(os.scandir(output_dir))


def mean_metric_over_samples(metric_tensor):
    if not th.is_tensor(metric_tensor):
        metric_tensor = th.as_tensor(metric_tensor)

    if metric_tensor.ndim != 2:
        raise ValueError(
            f"Expected metric tensor with shape [B, L], got {tuple(metric_tensor.shape)}"
        )

    if not th.is_floating_point(metric_tensor):
        metric_tensor = metric_tensor.to(dtype=th.float32)

    return metric_tensor.mean(dim=0)


def std_metric_over_samples(metric_tensor):
    if not th.is_tensor(metric_tensor):
        metric_tensor = th.as_tensor(metric_tensor)

    if metric_tensor.ndim != 2:
        raise ValueError(
            f"Expected metric tensor with shape [B, L], got {tuple(metric_tensor.shape)}"
        )

    if not th.is_floating_point(metric_tensor):
        metric_tensor = metric_tensor.to(dtype=th.float32)

    return metric_tensor.std(dim=0, unbiased=False)


def build_distribution_metric_plots(
    lens_out,
    layer_labels=None,
    shared_lens_source=None,
):
    metric_names = [
        "entropy",
        "top1_prob",
        "prob_margin",
    ]

    output_root = get_logit_lens_img_dir()
    os.makedirs(output_root, exist_ok=True)

    return plot_sample_mean_metric_panels(
        lens_out=lens_out,
        metric_names=metric_names,
        sample_reducer=mean_metric_over_samples,
        output_root=output_root,
        lens_source_name=shared_lens_source,
        layer_labels=layer_labels,
    )


def print_metric_markdown_tables(
    lens_out,
    layer_labels=None,
    shared_lens_source=None,
):
    metric_names = [
        "entropy",
        "top1_prob",
        "prob_margin",
    ]
    act_names = [
        "attn_resid",
        "mlp_resid",
    ]

    print("=== RAW METRIC TABLES (MARKDOWN) ===")

    for metric_name in metric_names:
        markdown_tables = format_metric_markdown_tables(
            lens_out=lens_out,
            metric_name=metric_name,
            sample_reducer=mean_metric_over_samples,
            std_reducer=std_metric_over_samples,
            lens_source_name=shared_lens_source,
            layer_labels=layer_labels,
            act_names=act_names,
        )

        for table in markdown_tables:
            print(table)
            print()


def build_generated_token_alignment_plots(
    lens_out,
    layer_labels=None,
    shared_lens_source=None,
):
    metric_names = [
        "surprisal",
        "target_rank",
    ]
    act_names = [
        "attn_resid",
        "mlp_resid",
    ]
    positions = [0.1, 0.5, 0.9, 0.95]

    output_root = get_logit_lens_img_dir()
    os.makedirs(output_root, exist_ok=True)

    return plot_alignment_metric_panels(
        lens_out=lens_out,
        metric_names=metric_names,
        sample_reducer=mean_metric_over_samples,
        output_root=output_root,
        lens_source_name=shared_lens_source,
        layer_labels=layer_labels,
        positions=positions,
        act_names=act_names,
    )


def main():
    device = "cuda" if th.cuda.is_available() else "cpu"
    act_modules = ["resid_pre", "attn_resid", "mlp_resid"]
    normalized_positions = [0.1, 0.5, 0.9, 0.95]
    layers = None
    lens_modes = ["native"]
    shared_lens_source = "rlvr"
    layer_labels = None
    batch_size = 10
    max_sample = 100
    sample_selection_seed = 42

    act_dataset = get_activation_dataset()
    h5_path = act_dataset["h5_path"]
    token_cache_path = resolve_token_cache_path(h5_path)

    activation_out, _layer_ids = load_sample_batch(
        batch_size=batch_size,
        model_names=None,
        act_modules=act_modules,
        layers=layers,
        h5_path=h5_path,
        max_sample=max_sample,
        sample_seed=sample_selection_seed,
    )
    model_names = list(activation_out.keys())
    layer_labels = [format_layer_group_label(layer_id) for layer_id in _layer_ids]
    base_model_name = model_names[0]
    sample_ids = sorted(activation_out[base_model_name][act_modules[0]].keys())

    expected_metadata = build_lens_out_metadata(
        h5_path=h5_path,
        token_cache_path=token_cache_path,
        positions=normalized_positions,
        act_modules=act_modules,
        lens_modes=lens_modes,
        shared_lens_source=shared_lens_source,
        max_sample=max_sample,
        batch_size=batch_size,
        sample_selection_seed=sample_selection_seed,
        layers=layers,
        layer_labels=layer_labels,
        model_names=model_names,
        sample_ids=sample_ids,
    )

    output_dir_populated = has_populated_lens_outputs()
    output_path = resolve_lens_output_path(lens_modes, shared_lens_source)

    lens_out = None

    if output_dir_populated and os.path.exists(output_path):
        print("=== LOADING EXISTING LENS OUT ===")
        cached_lens_out = th.load(output_path, map_location="cpu")

        if (
            validate_lens_out_cache(cached_lens_out, expected_metadata)
            and has_probability_margin_table_inputs(cached_lens_out)
        ):
            lens_out = cached_lens_out
        else:
            print("=== EXISTING LENS OUT CACHE INVALID: RECOMPUTING ===")

    if lens_out is None:
        if not output_dir_populated:
            print("=== OUTPUTS EMPTY OR MISSING: POPULATING LENS OUT ===")
        elif not os.path.exists(output_path):
            print("=== TARGET LENS OUT MISSING: POPULATING LENS OUT ===")
        model_comparison = create_model_comparison(model_names)
        bank_lens = build_bank_lens(
            model_names=model_names,
            device=device,
            shared_lens_source=shared_lens_source,
        )

        lens_out = lens_view(
            views=normalized_positions,
            act_names=act_modules,
            lens_modes=lens_modes,
            model_names=model_names,
            activation_out=activation_out,
            lens_bank=bank_lens,
            model_comparison=model_comparison,
            device=device,
            token_cache_path=token_cache_path,
        )
        lens_out = attach_lens_out_metadata(
            lens_out,
            h5_path=h5_path,
            token_cache_path=token_cache_path,
            positions=normalized_positions,
            act_modules=act_modules,
            lens_modes=lens_modes,
            max_sample=max_sample,
            batch_size=batch_size,
            sample_selection_seed=sample_selection_seed,
            layers=layers,
            layer_labels=layer_labels,
            model_names=model_names,
            sample_ids=sample_ids,
            shared_lens_source=shared_lens_source,
        )
        output_path = save_lens_out(
            lens_out=lens_out,
            lens_modes=lens_modes,
            shared_lens_source=shared_lens_source,
        )

    print("Available lens modes:", lens_modes)
    print("Shared lens source:", shared_lens_source)
    print("Computed positions:", infer_positions(lens_out))
    print("Saved lens_out to:", output_path)
    print("Resolved layer labels:", layer_labels)

    print("=== BUILDING LOGIT LENS PLOTS ===")
    plot_paths = build_distribution_metric_plots(
        lens_out,
        layer_labels=layer_labels,
        shared_lens_source=shared_lens_source,
    )
    print_metric_markdown_tables(
        lens_out,
        layer_labels=layer_labels,
        shared_lens_source=shared_lens_source,
    )
    alignment_plot_paths = build_generated_token_alignment_plots(
        lens_out,
        layer_labels=layer_labels,
        shared_lens_source=shared_lens_source,
    )
    print("Saved logit lens plots:")
    for plot_path in plot_paths:
        print(plot_path)
    print("Saved generated-token alignment plots:")
    for plot_path in alignment_plot_paths:
        print(plot_path)


if __name__ == "__main__":
    main()

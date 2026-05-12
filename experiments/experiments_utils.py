import os
from dataclasses import dataclass

import torch as th

from experiments.act_dataset_utils import format_layer_group_label, load_sample_batch


LENS_OUT_METADATA_KEY = "__metadata__"


@dataclass
class ActivationBatch:
    """Container for sampled activations and related labels."""

    activation_out: dict
    model_names: list
    layer_labels: list
    sample_ids: list


def normalize_metadata_value(value):
    """Convert tensors and tuples into cache-friendly Python values."""
    if th.is_tensor(value):
        return value.detach().cpu().tolist()

    if isinstance(value, tuple):
        return [normalize_metadata_value(item) for item in value]

    if isinstance(value, list):
        return [normalize_metadata_value(item) for item in value]

    return value


def abs_path_or_none(path):
    """Return an absolute path, preserving None."""
    if path is None:
        return None

    return os.path.abspath(path)


def attach_metadata(lens_out, metadata):
    """Attach metadata to a lens output dict without mutating the original."""
    lens_out = dict(lens_out)
    lens_out[LENS_OUT_METADATA_KEY] = metadata
    return lens_out


def metadata_matches(cached_metadata, expected_metadata):
    """Check cache metadata and print the first mismatch."""
    if not isinstance(cached_metadata, dict):
        return False

    for key, expected_value in expected_metadata.items():
        if cached_metadata.get(key) != expected_value:
            print(f"Cache mismatch on metadata field: {key}")
            print(f"cached   = {cached_metadata.get(key)}")
            print(f"expected = {expected_value}")
            return False

    return True


def resolve_token_cache_path(h5_path):
    """Infer the token cache path from an activation HDF5 path."""
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


def _as_list_or_none(value):
    if value is None:
        return None

    if isinstance(value, str):
        return [value]

    return list(value)


def _normalize_act_module_name(act_module):
    """Map experiment-friendly activation names to saved dataset names."""
    aliases = {
        "attn_out": "attn_out_act",
        "mlp_out": "mlp_out_act",
        # resid_mid names the residual stream after attention and before MLP.
        "resid_mid": "attn_resid",
    }
    return aliases.get(act_module, act_module)


def _normalize_act_modules(act_modules):
    """Normalize an activation module sequence while preserving None."""
    act_modules = _as_list_or_none(act_modules)

    if act_modules is None:
        return None

    return [
        _normalize_act_module_name(act_module)
        for act_module in act_modules
    ]


def _resolve_act_modules(config, act_modules=None):
    if act_modules is not None:
        return _normalize_act_modules(act_modules)

    if hasattr(config, "act_modules"):
        return _normalize_act_modules(config.act_modules)

    if hasattr(config, "patch_modules"):
        return _normalize_act_modules(config.patch_modules)

    raise AttributeError(
        "load_activation_batch requires act_modules, config.act_modules, "
        "or config.patch_modules."
    )


def _resolve_sample_id_module(activation_out, model_name, requested_modules):
    if requested_modules:
        return requested_modules[0]

    metadata_keys = {"prompt_len", "completion_len", "total_len"}
    for key in activation_out[model_name].keys():
        if key not in metadata_keys:
            return key

    raise ValueError("No activation modules found in activation_out.")


def load_activation_batch(config, h5_path, act_modules=None):
    """Load a sampled activation batch for lens, patching, or custom configs."""
    requested_modules = _resolve_act_modules(config=config, act_modules=act_modules)

    activation_out, layer_ids = load_sample_batch(
        batch_size=config.batch_size,
        model_names=None,
        act_modules=requested_modules,
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
    sample_id_module = _resolve_sample_id_module(
        activation_out=activation_out,
        model_name=base_model_name,
        requested_modules=requested_modules,
    )
    sample_ids = sorted(activation_out[base_model_name][sample_id_module].keys())

    return ActivationBatch(
        activation_out=activation_out,
        model_names=model_names,
        layer_labels=layer_labels,
        sample_ids=sample_ids,
    )

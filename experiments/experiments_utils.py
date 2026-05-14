import os
from dataclasses import dataclass

import torch as th

from experiments.act_dataset_utils import format_layer_group_label, load_sample_batch
from experiments.tok_dataset_utils import (
    load_token_cache,
    build_token_index,
)

OUT_METADA_KEY = "__metadata__"

ACT_MODULE_ALIASES = {
    # direct outputs
    "attn_out": "attn_out",
    "attn_out_act": "attn_out",
    "mlp_out": "mlp_out",
    "mlp_out_act": "mlp_out",

    # residual stream names
    "resid_pre": "resid_pre",
    "resid_pre_act": "resid_pre",
    "resid_mid": "resid_mid",
    "attn_resid": "resid_mid",
    "resid_post": "resid_post",
    "mlp_resid": "resid_post",
}

CANONICAL_TO_SAVED_ACT = {
    "attn_out": "attn_out_act",
    "mlp_out": "mlp_out_act",
    "resid_pre": "resid_pre",
    "resid_mid": "attn_resid",
    "resid_post": "mlp_resid",
}

CANONICAL_TO_HOOK_COMPONENT = {
    "attn_out": "attn_out",
    "mlp_out": "mlp_out",
    "resid_pre": "resid_pre",
    "resid_mid": "resid_mid",
    "resid_post": "resid_post",
}

@dataclass
class ActivationBatch:
    """Container for sampled activations and related labels."""

    activation_out: dict
    model_names: list
    layer_labels: list
    sample_ids: list

def normalize_act_module_name(act_module):
    act_module = str(act_module)

    if act_module not in ACT_MODULE_ALIASES:
        raise KeyError(
            f"Unknown activation module: {act_module}. "
            f"Supported names: {sorted(ACT_MODULE_ALIASES.keys())}"
        )

    return ACT_MODULE_ALIASES[act_module]


def resolve_saved_act_name(act_module):
    canonical_name = normalize_act_module_name(act_module)
    return CANONICAL_TO_SAVED_ACT[canonical_name]


def resolve_hook_component(act_module):
    canonical_name = normalize_act_module_name(act_module)
    return CANONICAL_TO_HOOK_COMPONENT[canonical_name]


def resolve_act_module_names(act_module):
    canonical_name = normalize_act_module_name(act_module)
    return {
        "canonical": canonical_name,
        "saved": CANONICAL_TO_SAVED_ACT[canonical_name],
        "hook_component": CANONICAL_TO_HOOK_COMPONENT[canonical_name],
    }

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
    lens_out[OUT_METADA_KEY] = metadata
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
    return resolve_saved_act_name(act_module)


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
        model_names=config.model_names,
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

def normalize_layer_label(layer):
    layer = str(layer)

    if layer.startswith("layer_"):
        layer = layer.removeprefix("layer_")
    if layer.isdigit():
        return str(int(layer))

    return layer

def build_layer_idx_map(layer_labels):
    return {
        normalize_layer_label(layer_label) : idx 
        for idx, layer_label in enumerate(layer_labels)
    }

def normalize_activation_layer_idx(current_layer, layer_labels):
    layer_idx_map = build_layer_idx_map(layer_labels)
    current_layer = normalize_layer_label(current_layer)

    if current_layer not in layer_idx_map:
        raise KeyError(
            f"Layer {current_layer} not found in saved activations. "
            f"Available layers: {list(layer_idx_map.keys())}"
        )

    return layer_idx_map[current_layer]

def return_token_cache(token_cache_path) :
    return load_token_cache(token_cache_path)

def return_token_index(h5_path): 
    token_cache_path = resolve_token_cache_path(h5_path)
    token_index = build_token_index(return_token_cache(token_cache_path))

    return token_index
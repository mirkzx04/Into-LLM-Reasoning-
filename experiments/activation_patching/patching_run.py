import os
import sys
import torch as th
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from experiments.tok_dataset_utils import (
    load_token_cache,
    build_token_index,
)

from experiments.act_dataset_utils import get_activation_dataset
from experiments.experiments_conf import PatchConfig
from experiments.experiments_utils import (
    load_activation_batch, 
    resolve_token_cache_path,
    abs_path_or_none,
    attach_metadata, 
    normalize_metadata_value
)
from experiments.activation_patching.pathcing_utils import patch_view

def build_metadata(
    h5_path, 
    token_cache_path,
    config,
    activation_batch
):
    PATCH_CACHE_SCHEMA_VERSION = "activation_patching_v1"

    return {
        "schema_version": PATCH_CACHE_SCHEMA_VERSION,
        "h5_path": abs_path_or_none(h5_path),
        "token_cache_path": abs_path_or_none(token_cache_path),
        "positions": normalize_metadata_value(config.positions),
        "patch_modules": normalize_metadata_value(config.patch_modules),
        "recipient_name": config.recipient_name,
        "donor_name": config.donor_name,
        "layers": normalize_metadata_value(config.layers),
        "batch_size": config.batch_size,
        "max_sample": config.max_sample,
        "sample_selection_seed": config.sample_selection_seed,
        "model_names": normalize_metadata_value(activation_batch.model_names),
        "layer_labels": normalize_metadata_value(activation_batch.layer_labels),
        "sample_ids": normalize_metadata_value(activation_batch.sample_ids),
    }

def save_patch_cache(
    config,
    patch_out,
):
    position_tag = "_".join(str(p).replace(".", "p") for p in config.positions)
    module_tag = "_".join(config.patch_modules)

    patch_cache_name = (
        f"patch_out_"
        f"recipient-{config.recipient_name}_"
        f"donor-{config.donor_name}_"
        f"modules-{module_tag}_"
        f"pos-{position_tag}.pt"
    )

    patch_cache_dir = os.path.join(
        project_root,
        "experiments",
        "activation_patching",
        "patch_cache",
    )
    patch_cache_path = os.path.join(patch_cache_dir, patch_cache_name)

    os.makedirs(os.path.dirname(patch_cache_path), exist_ok=True)
    th.save(patch_out, patch_cache_path)

def main():
    config = PatchConfig()
    act_dataset = get_activation_dataset()
    h5_path = act_dataset["h5_path"]

    activation_batch = load_activation_batch(
        config=config, 
        h5_path=h5_path,
        act_modules=config.patch_modules
    )

    token_cache_path = resolve_token_cache_path(h5_path)
    token_cache = load_token_cache(token_cache_path)
    token_index = build_token_index(token_cache)
    del token_cache

    patch_out = patch_view(
        norm_positions=config.positions,
        act_names=config.patch_modules,
        recivient_name=config.recipient_name,
        donor_name=config.donor_name,
        token_index=token_index,
        sample_ids = activation_batch.sample_ids,
        device = "cuda",
        layer_labels=activation_batch.layer_labels,
        activation_out=activation_batch.activation_out
    )    

    patch_out = attach_metadata(
        patch_out,
        build_metadata(
            h5_path=h5_path,
            token_cache_path=token_cache_path,
            config=config,
            activation_batch=activation_batch
        )
    )

    save_patch_cache(config=config, patch_out=patch_out)

    
if __name__ == "__main__":
    main()

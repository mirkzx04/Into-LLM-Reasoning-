import os
import sys
import torch as th
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from experiments.act_dataset_utils import get_activation_dataset
from experiments.experiments_conf import PatchConfig
from experiments.experiments_utils import (
    OUT_METADATA_KEY,
    load_activation_batch, 
    return_token_index,
    abs_path_or_none,
    attach_metadata, 
    normalize_metadata_value,
    resolve_token_cache_path,
    metadata_matches,
)
from experiments.activation_patching.pathcing_utils import patch_view, build_patching_name
from experiments.activation_patching.patch_plot import plot_requested_patch_metrics

REQUIRED_PATCH_METRICS = {
    "recovery_score",
    "delta_logit_diff",
    "logit_diff_recovery",
    "sample_ids",
    "activation_delta_norm",
    "activation_relative_delta_norm",
    "activation_cosine_similarity",
}

def has_required_patch_structure(patch_out, expected_metadata):
    """
    Check if the loaded patch_out structure respect the waiting structure
    """
    if not isinstance(patch_out, dict):
        print("Patch cache is not a dict")
        return False

    # Check if the pathing name key is correct or it is missed
    patching_name = build_patching_name(
        recivient_name=expected_metadata["recivient_name"],
        donor_name = expected_metadata["donor_name"]
    )
    if patching_name not in patch_out:
        print(f"Patch cache missing patchinbg key : {patching_name}")

    patching_ref = patch_out[patching_name]
    
    # Extract expected keys from expected metadata
    expected_layers = expected_metadata["layer_labels"]
    expected_act_names = expected_metadata["patch_modules"]
    expected_positions = expected_metadata["positions"]

    for layer in expected_layers: 
        if layer not in patching_ref:
            print(f"Patch cache missing layer : {layer}")
            return False
        
        for act_name in expected_act_names:
            if act_name not in patching_ref[layer]:
                print(f"Patch cache missing act_name={act_name} for layer={layer}")
                return False

            for position in expected_positions:
                if position not in patching_ref[layer][act_name]:
                    print(
                        "Patch cache missing position "
                        f"layer={layer}, act_name={act_name}, position={position}"
                    )
                    return False

                metrics_ref = patching_ref[layer][act_name][position]
                missing = REQUIRED_PATCH_METRICS.difference(metrics_ref.keys())

                if missing:
                    print(
                        "Patch cache missing required metrics "
                        f"layer={layer}, act_name={act_name}, position={position}, "
                        f"missing={sorted(missing)}"
                    )
                    return False

    return True

def validate_patch_out_cache(patch_out, expected_metadata) :
    cached_metadata = patch_out.get(OUT_METADATA_KEY)

    if not metadata_matches(
        cached_metadata=cached_metadata,
        expected_metadata=expected_metadata
    ): 
        return False
    
    if not has_required_patch_structure(
        patch_out=patch_out,
        expected_metadata=expected_metadata,
    ) : 
        return False
    
    return True

def instance_activation_batch(
    config,
    h5_path
): 
    return load_activation_batch(
        config=config, 
        h5_path=h5_path,
        act_modules=config.patch_modules
    )

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
        "recivient_name": config.recivient_name,
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
    position_tag,
    module_tag,
):
    """
    Build the patch cache name and its dir
    """

    patch_cache_name = (
        f"patch_out_"
        f"recivient-{config.recivient_name}_"
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

def execute_pathing(
    h5_path, 
    config, 
    token_index,
    token_cache_path,
    activation_batch,
    module_tag,
    positon_tag,
): 
    """
    Execute activation pathing for config model and return the patch cache
    """
    patch_out = patch_view(
        norm_positions=config.positions,
        act_names=config.patch_modules,
        recivient_name=config.recivient_name,
        donor_name=config.donor_name,
        token_index=token_index,
        sample_ids = activation_batch.sample_ids,
        device = "cuda",
        layer_labels=activation_batch.layer_labels,
        activation_out=activation_batch.activation_out
    )    

    # Attach metadata to trace patch cache 
    patch_out = attach_metadata(
        patch_out,
        build_metadata(
            h5_path=h5_path,
            token_cache_path=token_cache_path,
            config=config,
            activation_batch=activation_batch
        )
    )

    save_patch_cache(
        config=config, 
        patch_out=patch_out,
        position_tag=positon_tag,
        module_tag=module_tag
    ) # Save patch cache 

    return patch_out

def main():
    config = PatchConfig()
    act_dataset = get_activation_dataset()
    h5_path = act_dataset["h5_path"]

    position_tag = "_".join(str(p).replace(".", "p") for p in config.positions)
    module_tag = "_".join(config.patch_modules)
    
    patch_cache_path = os.path.join(
        project_root,
        "experiments",
        "activation_patching",
        "patch_cache",
        f"patch_out_recivient-{config.recivient_name}_"
        f"donor-{config.donor_name}_"
        f"modules-{module_tag}_"
        f"pos-{position_tag}.pt"
    )

    if os.path.exists(patch_cache_path):
        print("Validating patch cache metadata")
        expected_metada = build_metadata(
            h5_path=h5_path,
            token_cache_path=resolve_token_cache_path(h5_path),
            config=config,
            activation_batch=instance_activation_batch(config, h5_path)
        )
        patch_out = th.load(patch_cache_path, map_location = "cpu") # Load cached patch_out

        if validate_patch_out_cache(
            patch_out=patch_out,
            expected_metadata=expected_metada,
        ): 
            print("Plotting metrics in patch cache")
            plot_requested_patch_metrics(patch_out)

        else : 
            print("non-compliant metadata")
            patch_out = execute_pathing(
                h5_path=h5_path,
                config=config,
                token_index=return_token_index(h5_path),
                activation_batch=instance_activation_batch(config, h5_path),
                module_tag=module_tag,
                positon_tag=position_tag, 
                token_cache_path=resolve_token_cache_path(h5_path)
            )

            plot_requested_patch_metrics(patch_out)
    else :  
        print(f"=== Don't found a patch cache in {patch_cache_path}")
        patch_out = execute_pathing(
            h5_path=h5_path,
            config=config,
            token_index=return_token_index(h5_path),
            activation_batch=instance_activation_batch(config, h5_path),
            module_tag=module_tag,
            positon_tag=position_tag,
            token_cache_path=resolve_token_cache_path(h5_path)
        )
        plot_requested_patch_metrics(patch_out)

if __name__ == "__main__":
    main()

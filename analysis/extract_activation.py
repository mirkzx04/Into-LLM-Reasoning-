import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch as th
from transformer_lens import HookedTransformer
from tqdm import tqdm

from models.model import MODEL_ID, get_model, get_tokenizer
from analysis.generation import generate_reasoning
from analysis.organize_activation import ACTIVATION_BLOCKS, append_cache_to_single_h5

# Default context used for TransformerLens forward passes.
DEFAULT_N_CTX = 5_000

# Run all extraction on GPU when it is available.
DEVICE = "cuda" if th.cuda.is_available() else "cpu"

def forward_with_cache(model, input_tokens, positions, hook_names):
    # Run a forward pass and cache only the requested hooks.
    with th.no_grad():
        _, cache = model.run_with_cache(
            input_tokens,
            names_filter=hook_names,
            pos_slice=positions,
            return_type=None,
        )

    return cache

def get_hook_names(interesting_layers):
    # Build the exact hook names to cache for the selected layers.
    return [
        f"blocks.{layer}.{hook_name}"
        for layer in interesting_layers
        for hook_name in ACTIVATION_BLOCKS.values()
    ]

def get_activation_dataset_paths(
        save_path, 
        generator_name, 
        ood_dataset_name, 
        max_new_tokens
    ):    
    # Store the dataset directly inside save_path without nested folders.
    os.makedirs(save_path, exist_ok=True)
    dataset_name = f"{generator_name}_{ood_dataset_name}_max_{max_new_tokens}_acts"
    return {
        "dataset_name": dataset_name,
        "save_dir": save_path,
        "h5_path": os.path.join(save_path, f"{dataset_name}.h5"),
        "metadata_path": os.path.join(save_path, f"{dataset_name}_metadata.pt"),
    }

def load_existing_activation_dataset(h5_path, metadata_path):
    # Load the saved metadata when it exists and return the cached dataset info.
    metadata = None
    if os.path.exists(metadata_path):
        metadata = th.load(metadata_path, map_location="cpu")

    return {
        "h5_path": h5_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
        "num_samples": None if metadata is None else metadata.get("num_samples"),
        "from_cache": True,
    }

def get_activation_positions(prompt_len, completion_len):
    # Save all token states across prompt and completion.
    total_len = prompt_len + completion_len
    return list(range(total_len))


def build_sample_metadata(generated):
    # Convert the generated token dataset into a metadata structure per sample.
    metadata = []

    sample_ids = generated["sample_id"]
    prompt_texts = generated["prompt_text"]
    completion_texts = generated["completion_text"]
    prompt_tokens = generated["prompt_tokens"]
    completion_tokens = generated["completion_tokens"]
    ground_truths = generated.get("ground_truth", [None] * len(prompt_tokens))

    for sample_id, prompt_text, completion_text, prompt_token, completion_token, ground_truth in zip(
        sample_ids,
        prompt_texts,
        completion_texts,
        prompt_tokens,
        completion_tokens,
        ground_truths,
    ):
        prompt_len = prompt_token.shape[-1]
        completion_len = completion_token.shape[-1]
        positions = get_activation_positions(
            prompt_len=prompt_len,
            completion_len=completion_len,
        )
        full_tokens = th.cat([prompt_token, completion_token], dim=-1)

        metadata.append(
            {
                "sample_id": int(sample_id),
                "prompt_text": prompt_text,
                "completion_text": completion_text,
                "ground_truth": ground_truth,
                "prompt_tokens": prompt_token.cpu(),
                "completion_tokens": completion_token.cpu(),
                "full_tokens": full_tokens.cpu(),
                "prompt_len": int(prompt_len),
                "completion_len": int(completion_len),
                "total_len": int(prompt_len + completion_len),
                "positions": positions,
                "num_positions": len(positions),
            }
        )

    return metadata


def save_metadata(metadata, metadata_path, max_new_tokens, batch_size, do_sample):
    # Save a lightweight metadata file next to the HDF5 dataset.
    payload = {
        "max_new_tokens": max_new_tokens,
        "generation_batch_size": batch_size,
        "do_sample": do_sample,
        "num_samples": len(metadata),
        "activation_blocks": ACTIVATION_BLOCKS,
        "metadata": [
            {
                "sample_id": item["sample_id"],
                "prompt_text": item["prompt_text"],
                "completion_text": item["completion_text"],
                "ground_truth": item["ground_truth"],
                "prompt_len": item["prompt_len"],
                "completion_len": item["completion_len"],
                "total_len": item["total_len"],
                "num_positions": item["num_positions"],
            }
            for item in metadata
        ],
    }

    th.save(payload, metadata_path)
    return payload


def load_tl_model(model_pth, n_ctx=DEFAULT_N_CTX):
    # Build a TransformerLens model for one checkpoint variant.
    tl_model = HookedTransformer.from_pretrained_no_processing(
        MODEL_ID,
        hf_model=get_model(model_pth),
        tokenizer=get_tokenizer(model_pth),
        device=DEVICE,
        dtype=th.bfloat16,
        n_ctx=n_ctx,
    )

    # Keep the config aligned with the runtime context window.
    tl_model.cfg.n_ctx = n_ctx
    tl_model.eval().to(DEVICE)
    return tl_model


def get_interesting_layers(num_layers):
    # Extract only the first, middle and last layer.
    return [0, num_layers // 2, num_layers - 1]


def extract_activation(
    max_new_tokens,
    batch_size,
    gen_model,
    gen_tokenizer,
    gen_dataset,
    model_desc,
    save_path,
    generator_name,
    ood_dataset_name,
):
    # Reuse the existing activation dataset when it is already available locally.
    dataset_paths = get_activation_dataset_paths(
        save_path=save_path,
        max_new_tokens=max_new_tokens,
        generator_name=generator_name,
        ood_dataset_name=ood_dataset_name
    )
    h5_path = dataset_paths["h5_path"]
    metadata_path = dataset_paths["metadata_path"]

    if os.path.exists(h5_path):
        return load_existing_activation_dataset(
            h5_path=h5_path,
            metadata_path=metadata_path,
        )

    # Generate one shared completion set that will be replayed through all models.
    token_cache_prefix = f"{generator_name}_{ood_dataset_name}_max{max_new_tokens}_tok"
    generated = generate_reasoning(
        model=gen_model,
        tokenizer=gen_tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        batch_size=batch_size,
        dataset = gen_dataset,
        save_path=save_path,
        filename_prefix=token_cache_prefix,
    )

    # Free the generation model before the activation extraction loop starts.
    del gen_model
    if DEVICE == "cuda":
        th.cuda.empty_cache()

    # Build and persist the metadata before writing activations.
    metadata = build_sample_metadata(generated)
    metadata = [item for item in metadata if item["total_len"] <= DEFAULT_N_CTX]
    metadata_payload = save_metadata(
        metadata=metadata,
        metadata_path=metadata_path,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        do_sample=False,
    )

    for model_pth, model_name in tqdm(model_desc, desc="Extracting model activations"):
        # Load one model variant at a time to keep the memory footprint controlled.
        tl_model = load_tl_model(model_pth=model_pth, n_ctx=DEFAULT_N_CTX)
        interesting_layers = get_interesting_layers(tl_model.cfg.n_layers)
        hook_names = get_hook_names(interesting_layers)

        for item in metadata:
            # Move the full prompt+completion sequence to the active device.
            full_tokens = item["full_tokens"].unsqueeze(0).to(DEVICE)
            positions = item["positions"]

            # Reset peak stats only when CUDA is available.
            if DEVICE == "cuda":
                th.cuda.reset_peak_memory_stats()

            cache = forward_with_cache(
                model=tl_model,
                input_tokens=full_tokens,
                positions=positions,
                hook_names=hook_names,
            )

            # Append the cached activations to the shared HDF5 file.
            append_cache_to_single_h5(
                h5_path=h5_path,
                cache=cache,
                item=item,
                model_name=model_name,
                interesting_layers=interesting_layers,
            )

            # Release per-sample tensors before processing the next sample.
            del cache
            del full_tokens
            if DEVICE == "cuda":
                th.cuda.empty_cache()

        # Release the current model before loading the next one.
        del tl_model
        if DEVICE == "cuda":
            th.cuda.empty_cache()

    return {
        "h5_path": h5_path,
        "metadata_path": metadata_path,
        "metadata": metadata_payload,
        "num_samples": metadata_payload["num_samples"],
        "from_cache": False,
    }

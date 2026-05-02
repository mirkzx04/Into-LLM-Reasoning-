import os 
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch as th

from transformers import AutoConfig
from transformer_lens import HookedTransformer

from models.model import get_model, get_tokenizer, MODEL_ID
from analysis.generation import generate_reasoning
from analysis.organize_activation import init_activation_store, append_activation, ACTIVATION_BLOCKS

RLVR_PTH = "rlvr_model_math"
SFT_PTH =  "sftf_model_math"

device = "cuda" if th.cuda.is_available() else "cpu"

def forward_with_cache(model, inpt, positions, names_filter):
    with th.no_grad(): 
        _, cache = model.run_with_cache(
            inpt,
            names_filter=names_filter,
            pos_slice=positions,
            return_type=None,
        )
            
    return cache

def get_activation_position(prompt_len, completion_len):
    total_len = prompt_len + completion_len
    
    # All token states: prompt + completion.
    return list(range(0, total_len))

   
    
def build_metadata(generated) :
    metadata = []

    sample_ids = generated["sample_id"]
    prompt_texts = generated["prompt_text"]
    completion_texts = generated["completion_text"]
    prompt_tokens = generated["prompt_tokens"]
    completion_tokens = generated["completion_tokens"]
    ground_truths = generated.get("ground_truth", [None] * len(prompt_tokens))

    for sample_id, prompt_text, completion_text, p_token, c_token, gt in zip(
        sample_ids,
        prompt_texts,
        completion_texts,
        prompt_tokens,
        completion_tokens,
        ground_truths,
    ):
        prompt_len = p_token.shape[-1]
        completion_len = c_token.shape[-1]

        positions = get_activation_position(
            prompt_len=prompt_len,
            completion_len=completion_len,
        )

        full_tokens = th.cat([p_token, c_token], dim=-1)

        metadata.append({
            "sample_id": int(sample_id),
            "prompt_text": prompt_text,
            "completion_text": completion_text,
            "ground_truth": gt,

            "prompt_tokens": p_token.cpu(),
            "completion_tokens": c_token.cpu(),
            "full_tokens": full_tokens.cpu(),

            "prompt_len": int(prompt_len),
            "completion_len": int(completion_len),
            "total_len": int(prompt_len + completion_len),

            "positions": positions,
            "num_positions": len(positions),
        })

    return metadata

def extract_activation(max_new_tokens, batch_size):
    model_names = ["base", "sftt", "rlvr"]
    model_activations = None
    do_sample = batch_size > 1

    save_path = f"activation_dataset"

    # Generate reasoning 
    hf_rlvr = get_model(RLVR_PTH)
    rlvr_tokenizer = get_tokenizer(RLVR_PTH)

    generated = generate_reasoning(hf_rlvr, rlvr_tokenizer, max_new_tokens, do_sample, batch_size)
    del hf_rlvr
    th.cuda.empty_cache()

    # Build metadata
    metadata = build_metadata(generated)

    for model_pth, model_name in [(None, "base"), (SFT_PTH, "sftt"), (RLVR_PTH, "rlvr")]:
        tl_model =HookedTransformer.from_pretrained(
            MODEL_ID,
            hf_model=get_model(model_pth),
            tokenizer=get_tokenizer(model_pth),
            device=device,
            dtype=th.bfloat16
        ) # Istance transformer lens model
        num_layers = tl_model.cfg.n_layers

        if model_activations is None:
            model_activations = init_activation_store(model_names, num_layers)
        hook_names = [
             f"blocks.{l}.{hook_name}"
            for l in range(num_layers)
            for hook_name in ACTIVATION_BLOCKS.values()
        ]

        # Get activation
        for item in metadata:
            full_tokens = item["full_tokens"].unsqueeze(0).to(device)
            positions = item["positions"]

            cache = forward_with_cache(
                model=tl_model,
                inpt=full_tokens,
                positions=positions,
                names_filter=hook_names,
            )

            append_activation(
                cache=cache,
                store=model_activations,
                model_name=model_name,
                num_layers=num_layers,
            )

            del cache
            th.cuda.empty_cache()

        del tl_model
        th.cuda.empty_cache()

    if save_path is not None:
         th.save(
        {
            "max_new_tokens": max_new_tokens,
            "batch_size": batch_size,
            "do_sample": do_sample,
            "activation_blocks": ACTIVATION_BLOCKS,
            "metadata": metadata,
            "activations": model_activations,
        },
        save_path,
    )
         
extract_activation(3000, 50)
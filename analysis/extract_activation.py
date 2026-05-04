import os 
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from models.rope_theta import get_rope_theta

import pkg_resources
import torch as th

from transformer_lens import HookedTransformer
from tqdm import tqdm

from models.model import get_model, get_tokenizer, MODEL_ID
from analysis.generation import generate_reasoning
from analysis.organize_activation import append_cache_to_single_h5, ACTIVATION_BLOCKS, np

RLVR_PTH = "rlvr_model_math"
SFT_PTH =  "sftt_model_math"

SAVE_PATH = f"activation_dataset"


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

def save_activation_shard(
    cache, 
    item, 
    model_name, 
    num_layers, 
    save_dir,
    activation_dtype = th.float16    
): 
    model_dir = os.path.join(save_dir, model_name)
    os.makedirs(model_dir, exist_ok = True)

    sample_id = item["sample_id"]

    shard = {
        "sample_id": sample_id,
        "model_name": model_name,

        "prompt_text": item["prompt_text"],
        "completion_text": item["completion_text"],
        "ground_truth": item["ground_truth"],

        "prompt_len": item["prompt_len"],
        "completion_len": item["completion_len"],
        "total_len": item["total_len"],

        "positions": item["positions"],
        "num_positions": item["num_positions"],

        "prompt_tokens": item["prompt_tokens"],
        "completion_tokens": item["completion_tokens"],
        "full_tokens": item["full_tokens"],

        "activation_blocks": ACTIVATION_BLOCKS,
        "activations": {},
    }

    for l in range(num_layers):
        shard["activations"][l] = {}

        for block_name, hook_name in ACTIVATION_BLOCKS.items():
            key = f"blocks.{l}.{hook_name}"

            act = cache[key].detach().cpu()

            # cache shape: [1, num_positions, d_model]
            act = act.squeeze(0).to(activation_dtype)

            shard["activations"][l][block_name] = act

    shard_path = os.path.join(
        model_dir,
        f"sample_{sample_id:06d}.pt"
    )

    th.save(shard, shard_path)

    return shard_path

def extract_activation(max_new_tokens, batch_size):
    do_sample = batch_size > 1
    
    # Check if save path exist, if not create it
    save_dir = os.path.join(
        SAVE_PATH,
        f"max_new_tokens_{max_new_tokens}_genbs_{batch_size}"
    )
    os.makedirs(save_dir, exist_ok=True)

    h5_path = os.path.join(save_dir, "activation_dataset.h5")
    metadata_path = os.path.join(save_dir, "metadata.pt")

    # Generate reasoning 
    hf_rlvr = get_model(RLVR_PTH)
    rlvr_tokenizer = get_tokenizer(RLVR_PTH)

    generated = generate_reasoning(hf_rlvr, rlvr_tokenizer, max_new_tokens, do_sample, batch_size)
    del hf_rlvr
    th.cuda.empty_cache()

    # Build metadata
    metadata = build_metadata(generated)

    th.save(
        {
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
        },
        metadata_path,
    )

    model_desc = [(None, "base"), (SFT_PTH, "sftt"), (RLVR_PTH, "rlvr")]
    for model_pth, model_name in tqdm(model_desc, desc="Extracting model activation"):
        tl_model =HookedTransformer.from_pretrained_no_processing(
            MODEL_ID,
            hf_model=get_model(model_pth),
            tokenizer=get_tokenizer(model_pth),
            device=device,
            dtype=th.bfloat16,
            n_ctx = 5_000
        ) # Istance transformer lens model
        tl_model.cfg.n_ctx = 5_000 # Settings model context
        tl_model.eval().to(device)
        num_layers = tl_model.cfg.n_layers
        interesting_layers = [0, num_layers // 2, num_layers - 1]

        hook_names = [
             f"blocks.{l}.{hook_name}"
            for l in interesting_layers
            for hook_name in ACTIVATION_BLOCKS.values()
        ]

        # Get activation
        for item in metadata:
            full_tokens = item["full_tokens"].unsqueeze(0).to(device)
            positions = item["positions"]

            seq_len = full_tokens.shape[-1]
            if seq_len > tl_model.cfg.n_ctx:
                del full_tokens
                th.cuda.empty_cache()
                continue
            
            th.cuda.reset_peak_memory_stats()

            cache = forward_with_cache(
                model=tl_model,
                inpt=full_tokens,
                positions=positions,
                names_filter=hook_names,
            )

            append_cache_to_single_h5(
                h5_path=h5_path,
                cache = cache,
                item = item,
                model_name=model_name,
                interesting_layers=interesting_layers,
                actication_dtype=np.float16
            )

            del cache
            del full_tokens
            th.cuda.empty_cache()

        del tl_model
        th.cuda.empty_cache()

extract_activation(3000, 40)
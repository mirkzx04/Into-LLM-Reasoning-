import os 
import sys
import json
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from tqdm import tqdm

import torch as th

from MATH_logic.dataset_utils.dataset_splitting import build_ood_eval_dataset

def tensor_to_list(x):
    if isinstance(x, th.Tensor):
        return x.detach().cpu().tolist()
    return list(x)

def load_token_dataset(save_path, filename_prefix) : 
    pt_path = os.path.join(save_path, f"{filename_prefix}.pt")

    if os.path.exists(pt_path):
        return th.load(pt_path, map_location="cpu")

    return None

def save_token_dataset(gen_out, save_path, filename_prefix):
    os.makedirs(save_path, exist_ok=True)

    pt_path = os.path.join(save_path, f"{filename_prefix}.pt")
    jsonl_path = os.path.join(save_path, f"{filename_prefix}.jsonl")

    # Saving with torch
    th.save(gen_out, pt_path)

    # Saving with json
    n_samples = len(gen_out["sample_id"])

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for idx in range(n_samples):
            row = {
                "sample_id": gen_out["sample_id"][idx],
                "prompt_text": gen_out["prompt_text"][idx],
                "completion_text": gen_out["completion_text"][idx],
                "prompt_tokens": tensor_to_list(gen_out["prompt_tokens"][idx]),
                "completion_tokens": tensor_to_list(gen_out["completion_tokens"][idx]),
                "ground_truth": gen_out["ground_truth"][idx],
                "prompt_len": len(gen_out["prompt_tokens"][idx]),
                "completion_len": len(gen_out["completion_tokens"][idx]),
            }

            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "pt_path": pt_path,
        "jsonl_path": jsonl_path,
        "num_samples": n_samples,
    }
def slice_prompt_pad(prompt_ids, attention_mask):
    """
    prompt_ids : [B, prompt_padded_len]
    attention_mask : [B, prompt_padded_len]
    """
    out = []
    for row, mask in zip(prompt_ids, attention_mask):
        out.append(row[mask.bool()])

    return out

def slice_completion(completion_ids, eos_id, keep_eos = True):
    """
    completion_ids : [B, gen_len]
    """
    out = []

    for row in completion_ids: 
        # Identify the position of eos token
        eos_pos = (row == eos_id).nonzero(as_tuple = False)
        if len(eos_pos) > 0:
            end = eos_pos[0].item()
            if keep_eos:
                end += 1
            row = row[:end]

        out.append(row)
    return out

def get_optional_column(dataset, name, start, end, default=None):
    if hasattr(dataset, "column_names") and name in dataset.column_names:
        return dataset[name][start:end]

    return [default] * (end - start)

def generate_reasoning(
    model, 
    tokenizer, 
    max_new_tokens, 
    do_sample, 
    batch_size, 
    dataset,
    save_path,
    filename_prefix,
    force_generation = False
):
    
    # If token dataset already exists, skip generation and return it
    if not force_generation:
        cached_dataset = load_token_dataset(
            save_path=save_path,
            filename_prefix=filename_prefix,
        )

        if cached_dataset is not None:
            return cached_dataset
    gen_out = {}

    model = model.eval().to("cuda" if th.cuda.is_available() else "cpu")

    gen_out = {
        "sample_id": [],
        "prompt_text": [],
        "completion_text": [],
        "prompt_tokens": [],
        "completion_tokens": [],
        "ground_truth": [],
    }

    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating reasoning"):
        # Settings batch_indices 
        batch_end = min(i + batch_size, len(dataset))

        prompts = dataset["prompt"][i:batch_end]
        ground_truth = get_optional_column(dataset, "solution", i, batch_end, default=None)

        input_ids = tokenizer(
            prompts,
            return_tensors = "pt",
            padding = True,
            truncation = True,
            add_special_tokens = False
        ).to(model.device) # Shape : [B, seq_len]

        # Generate reasoning
        gen_kwargs = {
            "do_sample" : do_sample,
            "max_new_tokens" : max_new_tokens,
            "eos_token_id" : tokenizer.eos_token_id,
            "pad_token_id" : tokenizer.pad_token_id,
            "use_cache" : True,
        }
        if do_sample:
            gen_kwargs.update({
                "top_p": 0.95,
                "temperature": 0.7,
            })
        
        with th.inference_mode():
            output_ids = model.generate(
                **input_ids,
                **gen_kwargs,
            )

        prompts_padded_ids = input_ids["input_ids"]
        prompts_attention_mask = input_ids["attention_mask"]

        prompts_padded_len = prompts_padded_ids.shape[-1]
        completion_padded_ids = output_ids[:, prompts_padded_len:]

        prompt_rows = slice_prompt_pad(
            prompt_ids=prompts_padded_ids,
            attention_mask=prompts_attention_mask,
        )

        completion_rows = slice_completion(
            completion_ids=completion_padded_ids,
            eos_id=tokenizer.eos_token_id,
            keep_eos=False,
        )

        completion_texts = [
            tokenizer.decode(row, skip_special_tokens=True)
            for row in completion_rows
        ]

        gen_out["sample_id"].extend(list(range(i, batch_end)))
        gen_out["prompt_text"].extend(prompts)
        gen_out["completion_text"].extend(completion_texts)
        gen_out["prompt_tokens"].extend(prompt_rows)
        gen_out["completion_tokens"].extend(completion_rows)
        gen_out["ground_truth"].extend(ground_truth)

    save_info = save_token_dataset(
        gen_out = gen_out,
        save_path = save_path,
        filename_prefix = filename_prefix
    )
    gen_out["save_info"] = save_info

    return gen_out
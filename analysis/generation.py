import os 
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import torch as th

from MATH_logic.dataset_utils.dataset_splitting import build_ood_eval_dataset

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

def generate_reasoning(model, tokenizer, max_new_tokens, do_sample, batch_size):
    dataset = build_ood_eval_dataset(tokenizer=tokenizer, mode="rlvr")
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

    for i in range(0, len(dataset), batch_size):
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

    return gen_out

    return gen_out

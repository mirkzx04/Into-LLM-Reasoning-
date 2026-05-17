import torch as th

from itertools import product
from tqdm import tqdm

from experiments.tok_dataset_utils import (
    load_token_cache,
    build_token_index,
)

from experiments.Logit_Lens.lens_metrics import (
    topk_token_ids,
    target_token_stats,
    kl_divergence_logits,
    topk_jaccard,
)


def build_jobs(
    norm_positions,
    act_names,
    lens_modes,
):
    # Build one flat job per experimental configuration.
    jobs = []

    for position, act_name, lens_mode in product(norm_positions, act_names, lens_modes):
        jobs.append({
            "position": position,
            "act_name": act_name,
            "lens_mode": lens_mode,
        })

    return jobs

def resolve_predictive_completion_indices(
    position,
    prompt_len,
    completion_len,
    direction=1,
):
    if isinstance(position, str):
        if position != "last_input_token" :
            raise ValueError(f"Unknmow lens view : {position}")

        if completion_len < 1 : 
            raise ValueError( "completion_len must be >= 1 to use last_input_token with an in-range first completion target")

        act_idx = prompt_len - 1
        if direction == -1:
            target_idx = act_idx - 1
        else : 
            target_idx = prompt_len
        
        return None, act_idx, target_idx

    if completion_len < 2:
        raise ValueError("completion_len must be >= 2 to select a completion token with an in-range next token")

    # For next-token prediction within the completion, only the first
    max_predictive_rel_idx = completion_len - 2
    rel_idx = int(round(max_predictive_rel_idx * position))
    rel_idx = max(0, min(rel_idx, max_predictive_rel_idx))

    act_idx = prompt_len + rel_idx

    if direction == -1:
        target_idx = act_idx - 1
    else:
        target_idx = act_idx + 1

    return rel_idx, act_idx, target_idx


def build_lens_out(
    jobs,
    model_names,
    model_comparison,
):
    lens_out = {}

    for j in jobs:
        position = j["position"]
        act_name = j["act_name"]
        lens_mode = j["lens_mode"]

        if position not in lens_out:
            lens_out[position] = {}

        if act_name not in lens_out[position]:
            lens_out[position][act_name] = {}

        if lens_mode not in lens_out[position][act_name]:
            lens_out[position][act_name][lens_mode] = {}

        # Single-model metrics.
        for model_name in model_names:
            lens_out[position][act_name][lens_mode][model_name] = {
                "target_logprob": [],
                "target_rank" : [],
                "sample_ids": [],
            }

        # Pairwise comparison metrics.
        for m1, m2 in model_comparison:
            comp_key = f"{m1}_vs_{m2}"

            lens_out[position][act_name][lens_mode][comp_key] = {
                "kl_divergence": [],
                "topk_jaccard": [],
            }

    return lens_out


def append_model_metrics(
    target_logprob,
    target_rank,
    model_metrics,
    sid
):
    if target_logprob.ndim == 2 and target_logprob.shape[-1] == 1:
        target_logprob = target_logprob.squeeze(-1)
    if target_rank.ndim == 2 and target_rank.shape[-1] == 1:
        target_rank = target_rank.squeeze(-1)

    model_metrics["target_logprob"].append(target_logprob.detach().cpu())
    model_metrics["target_rank"].append(target_rank.detach().cpu())
    model_metrics["sample_ids"].append(int(sid))

    return model_metrics

def compute_comparison_metrics(
    model_comparison,
    sid_logits,
    sid_top_ids,
    lens_out,
    position,
    act_name,
    lens_key,
):
    for m1, m2 in model_comparison:
        comp_key = f"{m1}_vs_{m2}"

        # sid_logits[m]: [L, V]
        # sid_top_ids[m]: [L, K]
        kl = kl_divergence_logits(sid_logits[m1], sid_logits[m2])
        lens_out[position][act_name][lens_key][comp_key]["kl_divergence"].append(kl.detach().cpu())

        jaccard = topk_jaccard(sid_top_ids[m1], sid_top_ids[m2])
        lens_out[position][act_name][lens_key][comp_key]["topk_jaccard"].append(jaccard.detach().cpu())


def aggregation_stack(
    model_names,
    lens_out,
    position,
    act_name,
    lens_key,
    model_comparison,
):
    # Stack single-model metrics:
    # list of [L] -> [B, L]
    for model_name in model_names:
        metric_ref = lens_out[position][act_name][lens_key][model_name]

        for k, v in metric_ref.items():
            if k != "sample_ids" and len(v) > 0:
                metric_ref[k] = th.stack(v, dim=0)

    # Stack pairwise metrics:
    # list of [L] or scalar-compatible tensors -> [B, ...]
    for m1, m2 in model_comparison:
        comp_key = f"{m1}_vs_{m2}"
        comp_ref = lens_out[position][act_name][lens_key][comp_key]

        if len(comp_ref["kl_divergence"]) > 0:
            comp_ref["kl_divergence"] = th.stack(
                comp_ref["kl_divergence"],
                dim=0,
            )
            comp_ref["topk_jaccard"] = th.stack(
                comp_ref["topk_jaccard"],
                dim=0,
            )


def resolve_lens(
    lens_bank,
    lens_mode,
    model_name,
):
    # Native mode:
    # each model uses its own norm + lm_head.
    if lens_mode == "native":
        native_lenses = lens_bank["native"]
        return native_lenses[model_name]

    # Shared mode:
    # every model uses the same lens selected in lens_run.py.
    if lens_mode == "shared":
        return lens_bank["shared"]

    raise ValueError(f"Unknown lens_mode: {lens_mode}")


def apply_lens(active_lens, act):
    lens_param = next(active_lens.parameters(), None)
    lens_dtype = lens_param.dtype if lens_param is not None else act.dtype
    lens_device = lens_param.device if lens_param is not None else act.device

    act_for_lens = act.to(device=lens_device, dtype=lens_dtype)

    with th.no_grad():
        logits = active_lens(act_for_lens)

    return logits.to(dtype=th.float32)

def format_view(position):
    if position is None:
        return "Fallback"
    if isinstance(position, str):
        return position
    return f"{float(position)}:.2f"

def lens_view(
    views,
    act_names,
    lens_modes,
    model_names,
    activation_out,
    lens_bank,
    model_comparison,
    device,
    token_cache_path="activation_dataset/token_cache/qwen25_1.5b_rlvr_ood_eval_dataset_max2000_tok.pt",
):
    jobs = build_jobs(views, act_names, lens_modes)
    lens_out = build_lens_out(
        jobs=jobs,
        model_names=model_names,
        model_comparison=model_comparison,
    )

    token_cache = load_token_cache(token_cache_path)
    token_index = build_token_index(token_cache)

    base_model = model_names[0]
    sample_ids = sorted(activation_out[base_model][act_names[0]].keys())
    total_steps = len(jobs) * len(sample_ids)

    with tqdm(total=total_steps, desc="Logit Lens", unit="sample") as pbar:
        for j in jobs:
            position = j["position"]
            act_name = j["act_name"]
            lens_mode = j["lens_mode"]

            if position is not None and not isinstance(position, (float, int, str)):
                raise TypeError("position must be None, float, int or str")
            if not isinstance(act_name, str):
                raise TypeError("act_name must be a string")
            if not isinstance(lens_mode, str):
                raise TypeError("lens_mode must be a string")

            for sid in sample_ids:
                sid_logits = {}
                sid_top_ids = {}

                for model_name in model_names:
                    # activation_out stores one sample as [L, seq_len, d_model].
                    act = th.as_tensor(
                        activation_out[model_name][act_name][sid],
                        dtype=th.float32,
                        device=device,
                    )

                    active_lens = resolve_lens(
                        lens_bank=lens_bank,
                        lens_mode=lens_mode,
                        model_name=model_name,
                    )

                    _, seq_len, _ = act.shape

                    prompt_len = activation_out[model_name]["prompt_len"][sid]
                    completion_len = activation_out[model_name]["completion_len"][sid]

                    full_tokens = token_index[sid]["full_tokens"]

                    if position is not None:
                        # State view:
                        # select one completion token state t and evaluate the
                        # prediction of token t + 1. The last completion token is
                        # excluded because full_tokens has no in-range successor.
                        _, act_idx, target_idx = resolve_predictive_completion_indices(
                            position=position,
                            prompt_len=prompt_len,
                            completion_len=completion_len,
                            direction=1,
                        )
                        act = act[:, act_idx, :]  # [L, D]
                        logits = apply_lens(active_lens=active_lens, act=act)  # [L, V]
                        logits_for_target = logits[:, None, :]

                        # target_token_stats expects target_token_id with shape [T].
                        target_token_id = th.tensor(
                            [int(full_tokens[target_idx])],
                            dtype=th.long,
                            device=device,
                        )
                    else:
                        # Fallback view: use a fixed late-sequence activation.
                        act = act[:, seq_len - 10, :]  # [L, D]
                        logits = apply_lens(active_lens=active_lens, act=act)  # [L, V]

                        target_token_id = None
                        logits_for_target = None

                    # Single-model metrics from logits [L, V].
                    top_ids = topk_token_ids(logits=logits, top_k=20)
                    if target_token_id is not None :
                        target_logprob, target_rank = target_token_stats(
                            logits=logits_for_target,
                            target_ids=target_token_id,
                        )
                    else : 
                        n_layers = logits.shape[0]
                        target_logprob, target_rank = th.zeros(n_layers, device = device), th.zeros(n_layers, device=device)
                
                    sid_logits[model_name] = logits 
                    sid_top_ids[model_name] = top_ids

                    metric_ref = lens_out[position][act_name][lens_mode][model_name]
                    append_model_metrics(
                        target_logprob=target_logprob,
                        target_rank=target_rank,
                        model_metrics=metric_ref,
                        sid=sid
                    )

                compute_comparison_metrics(
                    model_comparison=model_comparison,
                    sid_logits=sid_logits,
                    sid_top_ids=sid_top_ids,
                    lens_out=lens_out,
                    position=position,
                    act_name=act_name,
                    lens_key=lens_mode,
                )
                pbar.update(1)

            aggregation_stack(
                model_names=model_names,
                lens_out=lens_out,
                position=position,
                act_name=act_name,
                lens_key=lens_mode,
                model_comparison=model_comparison,
            )
                      
    return lens_out

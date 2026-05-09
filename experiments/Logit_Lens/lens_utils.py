import torch as th

from itertools import product
from tqdm import tqdm

from analysis.activation_view import choose_slicing

from experiments.tok_dataset_utils import (
    load_token_cache,
    build_token_index,
)

from experiments.Logit_Lens.metrics import (
    logit_entropy,
    confidence_stats,
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
        jobs.append(
            {
                "position": position,
                "act_name": act_name,
                "lens_mode": lens_mode,
            }
        )

    return jobs


def slice_tokens(
    tokens,
    position,
    direction,
    prompt_len,
    completion_len,
):
    # Compute the same normalized completion position used for activations.
    rel_idx = int(round((completion_len - 1) * position))
    act_idx = prompt_len + rel_idx

    # direction=1 means:
    # activation at act_idx predicts token at act_idx + 1.
    if direction == -1:
        target_idx = act_idx - 1
    else:
        target_idx = act_idx + 1

    return tokens[target_idx]


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

        if not isinstance(position, float):
            raise TypeError("position must be a float")
        if not isinstance(act_name, str):
            raise TypeError("act_name must be a string")
        if not isinstance(lens_mode, str):
            raise TypeError("lens_mode must be a string")

        if position not in lens_out:
            lens_out[position] = {}

        if act_name not in lens_out[position]:
            lens_out[position][act_name] = {}

        if lens_mode not in lens_out[position][act_name]:
            lens_out[position][act_name][lens_mode] = {}

        # Single-model metrics.
        for model_name in model_names:
            lens_out[position][act_name][lens_mode][model_name] = {
                "entropy": [],
                "eff_vocab": [],
                "top1_prob": [],
                "top2_prob": [],
                "prob_margin": [],
                "logit_margin": [],
                "top1_token_id": [],
                "top2_token_id": [],
                "target_logprob": [],
                "target_prob": [],
                "surprisal": [],
                "target_rank": [],
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
    model_metrics,
    entropy,
    eff_vocab,
    top1_prob,
    top2_prob,
    prob_margin,
    logit_margin,
    top1_token_id,
    top2_token_id,
    target_logprob,
    target_prob,
    surprisal,
    target_rank,
    sid,
):
    # Each metric has shape [L] before aggregation.
    model_metrics["entropy"].append(entropy.detach().cpu())
    model_metrics["eff_vocab"].append(eff_vocab.detach().cpu())
    model_metrics["top1_prob"].append(top1_prob.detach().cpu())
    model_metrics["top2_prob"].append(top2_prob.detach().cpu())
    model_metrics["prob_margin"].append(prob_margin.detach().cpu())
    model_metrics["logit_margin"].append(logit_margin.detach().cpu())
    model_metrics["top1_token_id"].append(top1_token_id.detach().cpu())
    model_metrics["top2_token_id"].append(top2_token_id.detach().cpu())

    # target_token_stats returns [L, 1], so squeeze the singleton T dimension.
    model_metrics["target_logprob"].append(target_logprob.squeeze(-1).detach().cpu())
    model_metrics["target_prob"].append(target_prob.squeeze(-1).detach().cpu())
    model_metrics["surprisal"].append(surprisal.squeeze(-1).detach().cpu())
    model_metrics["target_rank"].append(target_rank.squeeze(-1).detach().cpu())

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
        jaccard = topk_jaccard(sid_top_ids[m1], sid_top_ids[m2])

        lens_out[position][act_name][lens_key][comp_key]["kl_divergence"].append(
            kl.detach().cpu()
        )
        lens_out[position][act_name][lens_key][comp_key]["topk_jaccard"].append(
            jaccard.detach().cpu()
        )


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
    # Match the activation dtype to the lens dtype before the forward pass.
    # Then cast logits back to float32 for stable metric computation.
    lens_dtype = next(active_lens.parameters()).dtype
    act_for_lens = act.to(dtype=lens_dtype)

    with th.no_grad():
        logits = active_lens(act_for_lens)

    return logits.to(dtype=th.float32)


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

            if not isinstance(position, float):
                raise TypeError("position must be a float")
            if not isinstance(act_name, str):
                raise TypeError("act_name must be a string")
            if not isinstance(lens_mode, str):
                raise TypeError("lens_mode must be a string")

            pbar.set_description(
                f"Logit Lens | {act_name} | {lens_mode} | pos={position:.2f}"
            )

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
                        # position_normalized_completion selects act[:, prompt_len + rel_idx, :].
                        # This state predicts the next token, so target is full_tokens[act_idx + 1].
                        act = choose_slicing(
                            slicing_mode="position_normalized_completion",
                            act=act,
                            prompt_len=prompt_len,
                            completion_len=completion_len,
                            normalized_pos=position,
                        )  # [L, D]

                        logits = apply_lens(active_lens=active_lens, act=act)  # [L, V]

                        target_id = slice_tokens(
                            tokens=full_tokens,
                            position=position,
                            direction=1,
                            prompt_len=prompt_len,
                            completion_len=completion_len,
                        )

                        # target_token_stats expects target_ids with shape [T].
                        # Here T=1 because we are analyzing one selected token position.
                        target_ids = th.tensor(
                            [target_id], dtype=th.long, device=device
                        )

                        # target_token_stats expects logits [L, T, V].
                        logits_for_target = logits[:, None, :]  # [L, 1, V]

                    else:
                        # Fallback view: use a fixed late-sequence activation.
                        # This keeps your original behavior.
                        act = act[:, seq_len - 10, :]  # [L, D]
                        logits = apply_lens(active_lens=active_lens, act=act)  # [L, V]

                        target_ids = None
                        logits_for_target = None

                    # Single-model metrics from logits [L, V].
                    entropy, eff_vocab = logit_entropy(logits=logits)

                    top_ids, _top_probs, top1_prob, top2_prob, prob_margin, logit_margin = confidence_stats(
                        logits=logits,
                        top_k=10,
                    )

                    if target_ids is not None:
                        target_logprob, target_prob, surprisal, target_rank = target_token_stats(
                            logits=logits_for_target,
                            target_ids=target_ids,
                        )
                    else:
                        # Same layer shape as entropy/top1 metrics.
                        n_layers = logits.shape[0]

                        target_logprob = th.zeros(n_layers, device=device)
                        target_prob = th.zeros(n_layers, device=device)
                        surprisal = th.zeros(n_layers, device=device)
                        target_rank = th.zeros(n_layers, device=device)

                    sid_logits[model_name] = logits
                    sid_top_ids[model_name] = top_ids

                    metrics_ref = lens_out[position][act_name][lens_mode][
                        model_name
                    ]

                    append_model_metrics(
                        metrics_ref,
                        entropy,
                        eff_vocab,
                        top1_prob,
                        top2_prob,
                        prob_margin,
                        logit_margin,
                        top_ids[..., 0],
                        top_ids[..., 1],
                        target_logprob,
                        target_prob,
                        surprisal,
                        target_rank,
                        sid,
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

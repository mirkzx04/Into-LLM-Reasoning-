import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

import torch as th

from itertools import product
from tqdm import tqdm

from models.model import get_rope_theta, load_tl_model

from experiments.Logit_Lens.lens_utils import resolve_predictive_completion_indices
from experiments.experiments_utils import (
    normalize_activation_layer_idx,
    normalize_layer_label,
    resolve_act_module_names
)
from experiments.activation_patching.patching_metrics import (
    compute_recovery_score,
    target_token_stats_patching,
    select_receiver_foil_token,
    compute_logit_difference,
    delta_logit_difference,
    compute_logit_diff_recovery,
)

def build_patching_name(recivient_name, donor_name):
    return f"recivient-{recivient_name}_donor-{donor_name}"

def build_jobs(norm_positions, act_names, layer_labels):
    jobs = []

    for position, act_name, layer_label in product(norm_positions, act_names, layer_labels):
        jobs.append({
            "position": position,
            "layer_labels" : layer_label,
            "act_name": act_name,

        })
    return jobs

def build_patch_out(jobs, recivient_name, donor_name):
    patch_out = {}
    patching_name = build_patching_name(recivient_name=recivient_name, donor_name=donor_name)

    if patching_name not in patch_out:
        patch_out[patching_name] = {}

    for j in jobs:
        position = j["position"]
        act_name = j["act_name"]
        layer = j["layer_labels"]

        if layer not in patch_out[patching_name]:
            patch_out[patching_name][layer] = {}

        if act_name not in patch_out[patching_name][layer]:
            patch_out[patching_name][layer][act_name] = {}

        if position not in patch_out[patching_name][layer][act_name]:
            patch_out[patching_name][layer][act_name][position] = {}
        
        patch_out[patching_name][layer][act_name][position] = {
            "recovery_score": [],
            "delta_logit_diff": [],
            "logit_diff_recovery": [],
            "sample_ids": [],
            "activation_delta_norm": [],
            "activation_relative_delta_norm": [],
            "activation_cosine_similarity": [],
        }

    return patch_out

def instance_model(recivient_name, device) :
    if recivient_name == "base":
        return load_tl_model(model_pth=None, device=device, n_ctx=5_000).to(device)
    else :
        return load_tl_model(model_pth=f"{recivient_name}_model_math", device=device, n_ctx=5_000).to(device)

def make_patch_hook(donor_act, patch_stats):
    donor_act = donor_act.detach()

    def patch_hook(receiver_act, hook):
        """
        Replace the activation of the receiver model with that of the donor model
        Calculate metrics to understand how geometrically different the activations of the two models are

        metrics : 
            activation_delta_norm -> Measures L2
            activation_relative_norm -> It measures how large the donor-recipient difference is compared to the recipient state norm.
            activation_cosine_sim -> Measures the directional alignment between recipient and donor activation.
        Args : 
            receiver_act : Activation of the receiver model  | Shape : [B, seq_len, d_model]
        """
        patched = receiver_act.clone()
        receiver_vec = patched[:, -1, :].detach()
        donor_vec = donor_act.to(
            device = patched.device,
            dtype = patched.dtype,
        )

        receiver_vec = receiver_vec.to(
            device=donor_vec.device,
            dtype=donor_vec.dtype,
        )
        receiver_vec_flat = receiver_vec.squeeze(0)

        delta = donor_vec - receiver_vec_flat

        patch_stats["activation_delta_norm"] = delta.norm().detach().cpu()
        patch_stats["activation_relative_delta_norm"] = (
            delta.norm() / (receiver_vec.norm() + 1e-12)
        ).detach().cpu()
        patch_stats["activation_cosine_sim"] = th.nn.functional.cosine_similarity(
            receiver_vec_flat.float(),
            donor_vec.float(),
            dim = 0,
        ).detach().cpu()

        patched[:, -1, :] = donor_vec
        return patched

    return patch_hook

def run_patched_forward(
    recivient_model,
    donor_act, 
    hook_name, 
    tokens_for_recivient
):
    patch_stats = {}
    patch_hook = make_patch_hook(donor_act, patch_stats)

    with th.no_grad():
        patched_logits = recivient_model.run_with_hooks(
            tokens_for_recivient,
            fwd_hooks = [(hook_name, patch_hook)]
        ) # Shape [B, seq_len, vocab_size]

    return patched_logits[:, -1, :], patch_stats

def run_donor_forward(
    donor_model,
    tokens
):
    with th.no_grad():
        donor_logits = donor_model(tokens)

    return donor_logits[:, -1, :]

def run_recivient_forward(
    recivient_model, 
    tokens,
):
    with th.no_grad():
        recivient_logits = recivient_model(tokens)
    
    return recivient_logits[:, -1, :]

def append_patch_out(
    recovery_score_value,
    delta_logit_diff_value,
    logit_diff_recovery_value,
    sid,
    patch_metrics,
    patch_stats,
):
    patch_metrics["recovery_score"].append(
        recovery_score_value.squeeze().detach().cpu()
    )
    patch_metrics["delta_logit_diff"].append(
        delta_logit_diff_value.squeeze().detach().cpu()
    )
    patch_metrics["logit_diff_recovery"].append(
        logit_diff_recovery_value.squeeze().detach().cpu()
    )

    patch_metrics["activation_delta_norm"].append(
        patch_stats["activation_delta_norm"].detach().cpu()
    )
    patch_metrics["activation_relative_delta_norm"].append(
        patch_stats["activation_relative_delta_norm"].detach().cpu()
    )
    patch_metrics["activation_cosine_similarity"].append(
        patch_stats["activation_cosine_sim"].detach().cpu()
    )

    patch_metrics["sample_ids"].append(int(sid))
    return patch_metrics

def compute_target_logprob(logits, target_token_id): 
    logits_logprob, _ = target_token_stats_patching(
        logits=logits, 
        target_token_id=target_token_id
    ) 

    return logits_logprob

def stack_patch_out_metrics(patch_out) :
    """
    Convert patch_out metric lists from list[tensor scalar] to tensor[n_samples].

    Preserves:
        - nested patch_out structure
        - sample_ids as list[int]
        - metadata key, if present
    """
    for patching_name, patching_ref in patch_out.items():
        if patching_name == "__metadata__":
            continue

        for layer, layer_ref in patching_ref.items():
            for act_name, act_ref in layer_ref.items():
                for position, metrics_ref in act_ref.items():
                    for metric_name, values in metrics_ref.items():
                        if metric_name == "sample_ids":
                            continue

                        if len(values) == 0:
                            metrics_ref[metric_name] = th.empty(0)
                            continue

                        metrics_ref[metric_name] = th.stack([
                            value.detach().cpu().reshape(())
                            if th.is_tensor(value)
                            else th.as_tensor(value).reshape(())
                            for value in values
                        ])

    return patch_out

def patch_view(
    norm_positions, 
    act_names, 
    recivient_name, 
    donor_name,
    token_index,
    sample_ids,
    activation_out,
    device,
    layer_labels,
):
    # Build jobs
    jobs = build_jobs(
        norm_positions=norm_positions, 
        act_names=act_names, 
        layer_labels=layer_labels
    )
    patch_out = build_patch_out(
        jobs=jobs, 
        recivient_name=recivient_name, 
        donor_name=donor_name
    )
    recivient_model = instance_model(recivient_name=recivient_name, device = device)
    donor_model = instance_model(recivient_name=donor_name, device = device)

    patching_name = build_patching_name(recivient_name=recivient_name, donor_name=donor_name)
    total_steps = len(jobs) * len(sample_ids)

    with tqdm(total = total_steps, desc = "Activation Patching", unit = "sample") as pbar :
        for j in jobs : 
            position = j["position"]
            layer = j["layer_labels"]
            act_name = j["act_name"]

            names = resolve_act_module_names(act_name)
            saved_act_name = names["saved"]
            hook_component = names["hook_component"]

            for sid in sample_ids:
                # Extract donor model activation and token of the same sample
                donor_act = activation_out[donor_name][saved_act_name][sid] # Shape : [L, seq_len, d_model]
                donor_act = th.as_tensor(donor_act, dtype=th.float32, device=device) # Shape : [seq_len, d_model]
                
                full_tokens = th.as_tensor(token_index[sid]["full_tokens"], dtype=th.long, device=device) # Shape : [seq_len]

                completion_len = activation_out[donor_name]["completion_len"][sid]
                prompt_len = activation_out[donor_name]["prompt_len"][sid]

                # Extract layer idx to slice the activation
                saved_layer_idx = normalize_activation_layer_idx(
                    current_layer=layer,
                    layer_labels=layer_labels
                )
                real_layer_id = int(normalize_layer_label(layer))

                # Build hook name
                hook_name = f"blocks.{real_layer_id}.hook_{hook_component}"

                # Slice activation and token on selected position
                _, act_idx, target_idx = resolve_predictive_completion_indices(
                    position = position,
                    prompt_len = prompt_len,
                    completion_len = completion_len,
                )
                donor_act = donor_act[saved_layer_idx, act_idx, :] # Extract activation on selected position | Shape : [d_model]
                tokens_for_recivient = full_tokens[:target_idx].unsqueeze(0) # Take token until selected position | Shape : [1, seq_len]
                target_token_id = full_tokens[target_idx]

                # Compute logits for donor model and recivient model
                patched_logits, patch_stats = run_patched_forward(
                    recivient_model=recivient_model.eval(),
                    hook_name=hook_name,
                    tokens_for_recivient=tokens_for_recivient,
                    donor_act=donor_act
                ) # Shape : [B, vocab_size]
                donor_logits = run_donor_forward(
                    donor_model = donor_model.eval(),
                    tokens = tokens_for_recivient,
                ) # Shape : [B, vocab_size]
                recivient_logits = run_recivient_forward(
                    recivient_model=recivient_model.eval(),
                    tokens = tokens_for_recivient,
                ) # Shape : [B, vocab_size]

                # Compute log-probability on model logits 
                patched_logprob = compute_target_logprob(
                    logits=patched_logits,
                    target_token_id=target_token_id,
                )
                donor_logprob = compute_target_logprob(
                    logits=donor_logits,
                    target_token_id=target_token_id,
                )
                recipient_logprob = compute_target_logprob(
                    logits=recivient_logits,
                    target_token_id=target_token_id,
                )

                recovery_score_value = compute_recovery_score(
                    patched_score=patched_logprob,
                    donor_score=donor_logprob,
                    receiver_score=recipient_logprob,
                )

                foil_token_id = select_receiver_foil_token(
                    logits=recivient_logits,
                    target_token_id=target_token_id,
                )

                recipient_ld = compute_logit_difference(
                    logits=recivient_logits,
                    target_token_id=target_token_id,
                    foil_token_id=foil_token_id,
                )
                patched_ld = compute_logit_difference(
                    logits=patched_logits,
                    target_token_id=target_token_id,
                    foil_token_id=foil_token_id,
                )
                donor_ld = compute_logit_difference(
                    logits=donor_logits,
                    target_token_id=target_token_id,
                    foil_token_id=foil_token_id,
                )

                delta_logit_diff_value = delta_logit_difference(
                    patched_ld=patched_ld,
                    receiver_ld=recipient_ld,
                )
                logit_diff_recovery_value = compute_logit_diff_recovery(
                    patched_ld=patched_ld,
                    receiver_ld=recipient_ld,
                    donor_ld=donor_ld,
                )

                patch_metrics = patch_out[patching_name][layer][act_name][position]
                append_patch_out(
                    recovery_score_value=recovery_score_value,
                    delta_logit_diff_value=delta_logit_diff_value,
                    logit_diff_recovery_value=logit_diff_recovery_value,
                    sid=sid,
                    patch_metrics=patch_metrics,
                    patch_stats=patch_stats,
                )
                pbar.update(1)

    patch_out = stack_patch_out_metrics(patch_out)

    th.cuda.empty_cache()

    return patch_out



            










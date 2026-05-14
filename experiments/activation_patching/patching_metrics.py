import torch as th

def compute_recovery_score(
    patched_score,
    recivier_score,
    donor_score,
):
    """
    calculates recovery to understand when the specific activation of the donor 
    model, inserted into the receiver model, causally shifts the receiver model 
    towards the behavior of the donor model

    0 = The patch changes nothing
    1 = The patch completely restores the donor's behavior
    >1 = Overshoot
    <0 = The patch pushes in the opposite direction

    Args : 
        patched_score : Patched log_softmax score | Shape : []
        recivier_score : Recivier log_softmax score | Shape : []
        donor_score : Donor log_softmax score | Shape : []
    """

    num = patched_score - recivier_score
    denom = (donor_score - recivier_score) + 1e-12
    return num / denom

def delta_kl_divergence(
    kl_recivier_donor,
    kl_patched_donor,
) : 
    """
    Measures how much the patch shifts the complete distribution of the 
    recivier towards that of the donor

    Args : 
        kl_recivier_donor : KL(recivier_score, donor_score) | Shape : 
        kl_patched_donor : KL(patched_score, donor_score) | Shape : 
    """
    return (kl_recivier_donor - kl_patched_donor)

def target_token_stats_patching(logits, target_token_id):
    """
    logits:
        [V] or [B, V]
    target_token_id:
        scalar or [B]

    returns:
        target_logprob: scalar or [B]
        target_rank: scalar or [B]
    """

    original_was_1d = logits.ndim == 1

    if logits.ndim == 1:
        logits = logits[None, :]      # [1, V]

    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [V] or [B, V], got {tuple(logits.shape)}")

    if not th.is_tensor(target_token_id):
        target_token_id = th.tensor(target_token_id, device=logits.device)

    target_token_id = target_token_id.to(logits.device)

    if target_token_id.ndim == 0:
        target_token_id = target_token_id[None]  # [1]

    if target_token_id.ndim != 1:
        raise ValueError(
            f"target_token_id must be scalar or [B], got {tuple(target_token_id.shape)}"
        )

    if target_token_id.shape[0] != logits.shape[0]:
        raise ValueError(
            f"target_token_id batch must match logits batch: "
            f"{target_token_id.shape[0]} vs {logits.shape[0]}"
        )

    target_ids_exp = target_token_id[:, None]  # [B, 1]

    log_probs = logits.log_softmax(dim=-1)
    target_logprob = log_probs.gather(-1, target_ids_exp).squeeze(-1)  # [B]

    target_logits = logits.gather(-1, target_ids_exp).squeeze(-1)      # [B]
    target_rank = (logits > target_logits[:, None]).sum(dim=-1) + 1    # [B]

    if original_was_1d:
        target_logprob = target_logprob.squeeze(0)
        target_rank = target_rank.squeeze(0)

    return target_logprob, target_rank

import torch as th


def target_in_topk_hit(logits, target_ids, top_k=10):
    """
    Check whether the target token is inside the model top-k.

    logits: [B, V] or [V]
    target_ids: [B] or scalar-compatible

    returns:
        hit: [B] float tensor in {0.0, 1.0}
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    if logits.ndim != 2:
        raise ValueError(f"logits must have shape [B, V] or [V], got {tuple(logits.shape)}")

    if not th.is_tensor(target_ids):
        target_ids = th.as_tensor(target_ids, device=logits.device)

    if target_ids.ndim == 0:
        target_ids = target_ids.unsqueeze(0)

    if target_ids.ndim != 1:
        raise ValueError(f"target_ids must have shape [B] or scalar, got {tuple(target_ids.shape)}")

    if target_ids.shape[0] != logits.shape[0]:
        raise ValueError(
            f"target_ids batch size must match logits batch size: "
            f"{target_ids.shape[0]} vs {logits.shape[0]}"
        )

    target_ids = target_ids.to(device=logits.device, dtype=th.long)
    top_ids = logits.topk(k=top_k, dim=-1).indices  # [B, K]

    hit = (top_ids == target_ids[:, None]).any(dim=-1).to(dtype=th.float32)
    return hit


def delta_target_in_topk_hit(patched_logits, receiver_logits, target_ids, top_k=10):
    """
    Delta Top-K Hit:
        1[target in top-k(patched)] - 1[target in top-k(receiver)]

    patched_logits: [B, V] or [V]
    receiver_logits: [B, V] or [V]
    target_ids: [B] or scalar-compatible

    returns:
        delta_hit: [B] float tensor in {-1.0, 0.0, 1.0}
    """
    patched_hit = target_in_topk_hit(
        logits=patched_logits,
        target_ids=target_ids,
        top_k=top_k,
    )
    receiver_hit = target_in_topk_hit(
        logits=receiver_logits,
        target_ids=target_ids,
        top_k=top_k,
    )

    return patched_hit - receiver_hit
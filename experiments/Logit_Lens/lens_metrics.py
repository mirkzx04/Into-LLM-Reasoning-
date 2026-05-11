import torch as th


def topk_token_ids(logits, top_k=10):
    """
    Extract top-k token ids from logits.

    logits: [L, V] or [L, T, V]
    returns:
        top_ids: [L, K] or [L, T, K]
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    top_ids = logits.topk(k=top_k, dim=-1).indices
    return top_ids


def target_token_stats(logits, target_ids):
    """
    Compute primitive target-token alignment metrics.

    logits: [L, T, V]
    target_ids: [T]

    returns:
        target_logprob: [L, T]
        target_rank: [L, T]
    """
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [L, T, V], got {tuple(logits.shape)}")

    L, T, _ = logits.shape

    if target_ids.ndim != 1:
        raise ValueError(f"target_ids must have shape [T], got {tuple(target_ids.shape)}")

    if target_ids.shape[0] != T:
        raise ValueError(
            f"target_ids length must match T: got {target_ids.shape[0]} vs {T}"
        )

    target_ids = target_ids.to(logits.device)
    target_ids_exp = target_ids[None, :, None].expand(L, T, 1)

    log_probs = logits.log_softmax(dim=-1)

    target_logprob = log_probs.gather(-1, target_ids_exp).squeeze(-1)

    target_logits = logits.gather(-1, target_ids_exp).squeeze(-1)
    target_rank = (logits > target_logits[..., None]).sum(dim=-1) + 1

    return target_logprob, target_rank


def kl_divergence_logits(logits_p, logits_q):
    """
    Compute symmetric DKL between two logit distributions.

    logits_p, logits_q: [L, V] or [L, T, V]
    returns:
        dkl: [L] or [L, T]
    """
    if logits_p.shape != logits_q.shape:
        raise ValueError(
            f"logits_p and logits_q must have same shape: "
            f"{tuple(logits_p.shape)} vs {tuple(logits_q.shape)}"
        )

    log_p = logits_p.log_softmax(dim=-1)
    log_q = logits_q.log_softmax(dim=-1)

    p = log_p.exp()
    q = log_q.exp()
    kl_pq = (p * (log_p - log_q)).sum(dim=-1)
    kl_qp = (q * (log_q - log_p)).sum(dim=-1)
    dkl = 0.5 * (kl_pq + kl_qp)

    return dkl


def topk_jaccard(top_ids_a, top_ids_b):
    """
    Compute Jaccard similarity between two top-k token sets.

    top_ids_a, top_ids_b: [L, K] or [L, T, K]
    returns:
        jaccard: [L] or [L, T]
    """
    if top_ids_a.shape != top_ids_b.shape:
        raise ValueError(
            f"top_ids_a and top_ids_b must have same shape: "
            f"{tuple(top_ids_a.shape)} vs {tuple(top_ids_b.shape)}"
        )

    K = top_ids_a.shape[-1]

    matches = top_ids_a.unsqueeze(-1) == top_ids_b.unsqueeze(-2)

    intersection = matches.any(dim=-1).sum(dim=-1).float()
    union = 2 * K - intersection

    return intersection / union

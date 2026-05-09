import torch as th


def logit_entropy(logits):
    """
    Measures how concentrated or uncertain the distribution predicted by a layer is.

    logits: [L, T, V]
    """
    log_probs = logits.log_softmax(dim=-1)
    probs = log_probs.exp()

    entropy = -(probs * log_probs).sum(dim=-1)  # [L, T]
    eff_vocab = entropy.exp()                   # [L, T]

    return entropy, eff_vocab


def confidence_stats(logits, top_k=20):
    """
    Measures how dominant the most likely token is.

    logits: [L, T, V]
    """
    if top_k < 2:
        raise ValueError("top_k must be >= 2")

    log_probs = logits.log_softmax(dim=-1)

    top_logits, top_ids = logits.topk(k=top_k, dim=-1)
    top_log_probs = log_probs.gather(-1, top_ids)
    top_probs = top_log_probs.exp()

    top1_prob = top_probs[..., 0]                # [L, T]
    top2_prob = top_probs[..., 1]                # [L, T]
    prob_margin = top1_prob - top2_prob          # [L, T]
    logit_margin = top_logits[..., 0] - top_logits[..., 1]

    return top_ids, top_probs, top1_prob, top2_prob, prob_margin, logit_margin


def target_token_stats(logits, target_ids):
    """
    Measures how much the layer already predicts the real generated token.

    logits: [L, T, V]
    target_ids: [T]
    """
    L, T, V = logits.shape

    target_ids = target_ids.to(logits.device)
    target_ids_exp = target_ids[None, :, None].expand(L, T, 1)

    log_probs = logits.log_softmax(dim=-1)

    target_logprob = log_probs.gather(-1, target_ids_exp).squeeze(-1)
    target_prob = target_logprob.exp()
    surprisal = -target_logprob

    target_logits = logits.gather(-1, target_ids_exp).squeeze(-1)
    target_rank = (logits > target_logits[..., None]).sum(dim=-1) + 1

    return target_logprob, target_prob, surprisal, target_rank


def kl_divergence_logits(logits_p, logits_q):
    """
    KL(P || Q).

    logits_p, logits_q: [L, T, V]
    """
    log_p = logits_p.log_softmax(dim=-1)
    log_q = logits_q.log_softmax(dim=-1)

    p = log_p.exp()

    kl = (p * (log_p - log_q)).sum(dim=-1)  # [L, T]
    return kl

def topk_jaccard(top_ids_a, top_ids_b):
    """
    Measures whether two models/layers have the same top candidates.

    top_ids_a, top_ids_b: [L, T, K]
    """
    K = top_ids_a.shape[-1]

    matches = top_ids_a.unsqueeze(-1) == top_ids_b.unsqueeze(-2)

    intersection = matches.any(dim=-1).sum(dim=-1).float()
    union = 2 * K - intersection

    return intersection / union

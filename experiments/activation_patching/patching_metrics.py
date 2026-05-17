import torch as th

EPS = 1e-12


def _ensure_2d_logits(logits):
    original_was_1d = logits.ndim == 1

    if original_was_1d:
        logits = logits.unsqueeze(0)

    if logits.ndim != 2:
        raise ValueError(
            f"logits must have shape [V] or [B, V], got {tuple(logits.shape)}"
        )

    return logits, original_was_1d


def _ensure_1d_token_ids(token_ids, device, batch_size):
    if not th.is_tensor(token_ids):
        token_ids = th.tensor(token_ids, device=device)

    token_ids = token_ids.to(device)

    if token_ids.ndim == 0:
        token_ids = token_ids.unsqueeze(0)

    if token_ids.ndim != 1:
        raise ValueError(
            f"token_ids must be scalar or [B], got {tuple(token_ids.shape)}"
        )

    if token_ids.shape[0] != batch_size:
        raise ValueError(
            f"token_ids batch must match logits batch: "
            f"{token_ids.shape[0]} vs {batch_size}"
        )

    return token_ids


def gather_token_scores(logits, token_ids):
    logits, original_was_1d = _ensure_2d_logits(logits)
    token_ids = _ensure_1d_token_ids(token_ids, logits.device, logits.shape[0])

    values = logits.gather(-1, token_ids[:, None]).squeeze(-1)

    if original_was_1d:
        values = values.squeeze(0)

    return values


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
    logits, original_was_1d = _ensure_2d_logits(logits)
    target_token_id = _ensure_1d_token_ids(
        target_token_id,
        logits.device,
        logits.shape[0],
    )

    log_probs = logits.log_softmax(dim=-1)
    target_logprob = log_probs.gather(-1, target_token_id[:, None]).squeeze(-1)

    target_logits = logits.gather(-1, target_token_id[:, None]).squeeze(-1)
    target_rank = (logits > target_logits[:, None]).sum(dim=-1) + 1

    if original_was_1d:
        target_logprob = target_logprob.squeeze(0)
        target_rank = target_rank.squeeze(0)

    return target_logprob, target_rank


def select_receiver_foil_token(logits, target_token_id):
    """
    Choose the foil as the best non-target token of the unpatched receiver.
    """
    logits, original_was_1d = _ensure_2d_logits(logits)
    target_token_id = _ensure_1d_token_ids(
        target_token_id,
        logits.device,
        logits.shape[0],
    )

    masked_logits = logits.clone()
    masked_logits.scatter_(1, target_token_id[:, None], float("-inf"))
    foil_token_id = masked_logits.argmax(dim=-1)

    if original_was_1d:
        foil_token_id = foil_token_id.squeeze(0)

    return foil_token_id


def compute_recovery_score(patched_score, receiver_score, donor_score):
    """
    Recovery on target log-probability.
    """
    num = patched_score - receiver_score
    denom = (donor_score - receiver_score) + EPS
    return num / denom


def compute_logit_difference(logits, target_token_id, foil_token_id):
    target_logit = gather_token_scores(logits, target_token_id)
    foil_logit = gather_token_scores(logits, foil_token_id)
    return target_logit - foil_logit


def delta_logit_difference(patched_ld, receiver_ld):
    return patched_ld - receiver_ld


def compute_logit_diff_recovery(patched_ld, receiver_ld, donor_ld):
    num = patched_ld - receiver_ld
    denom = (donor_ld - receiver_ld) + EPS
    return num / denom
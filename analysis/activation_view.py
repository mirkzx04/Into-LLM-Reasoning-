import numpy as np
import torch as th


def _is_torch(x):
    return th.is_tensor(x)


def _mean_tokens(x):
    if _is_torch(x):
        return x.mean(dim=1)
    return x.mean(axis=1)


def _clamp(v, lo, hi):
    return max(lo, min(v, hi))


def choose_slicing(
    slicing_mode,
    act,
    prompt_len,
    completion_len,
    last_k=8,
    boundary_left=2,
    boundary_right=3,
    normalized_pos=0.5,
):
    """
    Args:
        act: [L, seq_len, d_model]   (numpy.ndarray or torch.Tensor)
        prompt_len: int
        completion_len: int
        last_k: number of final tokens of the completion to take
        boundary_left: how many tokens to take to the left of the prompt/completion boundary
        boundary_right: how many tokens to take to the right of the prompt/completion boundary
        normalized_pos: relative position in the completion in [0, 1]
    Returns:
        different modes return different shapes:
        - single token: [L, d_model]
        - token span: [L, T, d_model]
        - mean pooling: [L, d_model]
        - all_completion_flat: [L * T, d_model]
    """

    mode = slicing_mode.lower().strip()

    total_len = prompt_len + completion_len
    if total_len != act.shape[1]:
        raise ValueError(
            f"Inconsistent lengths: act.shape[1]={act.shape[1]} "
            f"but prompt_len + completion_len = {total_len}"
        )

    if prompt_len <= 0:
        raise ValueError("prompt_len must be > 0")

    if completion_len <= 0 and mode != "prompt_mean":
        raise ValueError("completion_len must be > 0 for this slicing mode")

    completion_start = prompt_len
    completion_end = prompt_len + completion_len          # exclusive
    last_input_idx = prompt_len - 1
    first_completion_idx = completion_start
    last_completion_idx = completion_end - 1

    if mode == "last_input_token":
        return act[:, last_input_idx, :]                  # [L, D]

    elif mode == "first_completion_token":
        return act[:, first_completion_idx, :]            # [L, D]

    elif mode == "last_completion_token":
        return act[:, last_completion_idx, :]             # [L, D]

    elif mode == "completion_mean":
        return _mean_tokens(act[:, completion_start:completion_end, :])   # [L, D]

    elif mode == "prompt_mean":
        return _mean_tokens(act[:, :prompt_len, :])      # [L, D]

    elif mode == "predictive_completion_mean":
        # positions predicting completion tokens
        pred_start = prompt_len - 1
        pred_end = prompt_len + completion_len - 1       # exclusive
        return _mean_tokens(act[:, pred_start:pred_end, :])              # [L, D]

    elif mode == "last_k_completion_tokens":
        k = min(last_k, completion_len)
        return act[:, completion_end - k:completion_end, :]              # [L, k, D]

    elif mode == "boundary_window":
        start = _clamp(prompt_len - boundary_left, 0, total_len)
        end = _clamp(prompt_len + boundary_right, 0, total_len)
        if end <= start:
            raise ValueError(f"Empty boundary window: start={start}, end={end}")
        return act[:, start:end, :]                                      # [L, T, D]

    elif mode == "all_completion_tokens":
        return act[:, completion_start:completion_end, :]                # [L, T, D]

    elif mode == "all_completion_flat":
        comp = act[:, completion_start:completion_end, :]                # [L, T, D]
        L, T, D = comp.shape
        if _is_torch(comp):
            return comp.reshape(L * T, D)                                # [L*T, D]
        return comp.reshape(L * T, D)                                    # [L*T, D]

    elif mode == "position_normalized_completion":
        # normalized_pos in [0,1]:
        # 0.0 = first completion token
        # 0.5 = approximately middle token
        # 1.0 = last completion token
        rel_idx = int(round((completion_len - 1) * normalized_pos))
        rel_idx = _clamp(rel_idx, 0, completion_len - 1)
        abs_idx = completion_start + rel_idx
        return act[:, abs_idx, :]                                        # [L, D]

    else:
        raise ValueError(f"Unknown slicing_mode: {slicing_mode}")
import torch as th

def to_list(x):
    """
    Convert tensor/list/tuple/scalar-like objects to plain Python lists.
    """
    if th.is_tensor(x):
        return x.detach().cpu().tolist()
    if isinstance(x, tuple):
        return list(x)
    return x


def load_token_cache(token_cache_path):
    """
    Load the .pt token cache saved during activation extraction.
    """
    return th.load(token_cache_path, map_location="cpu")


def build_token_index(token_cache):
    sample_ids = to_list(token_cache["sample_id"])
    prompt_tokens = to_list(token_cache["prompt_tokens"])
    completion_tokens = to_list(token_cache["completion_tokens"])

    token_index = {}

    for i, sid in enumerate(sample_ids):
        sid = int(sid)

        p_tokens = to_list(prompt_tokens[i])
        c_tokens = to_list(completion_tokens[i])

        token_index[sid] = {
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "full_tokens": p_tokens + c_tokens,
        }

    return token_index


def get_prompt_tokens(token_index, sid):
    """
    Return prompt token ids for one sample.
    """
    return token_index[int(sid)]["prompt_tokens"]


def get_completion_tokens(token_index, sid):
    """
    Return completion token ids for one sample.
    """
    return token_index[int(sid)]["completion_tokens"]


def get_full_tokens(token_index, sid):
    """
    Return prompt + completion token ids for one sample.
    """
    return token_index[int(sid)]["full_tokens"]


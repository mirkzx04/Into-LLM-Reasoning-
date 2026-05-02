import torch as th

ACTIVATION_BLOCKS = {
    "mlp_out_act": "hook_mlp_out",
    "attn_out_act": "hook_attn_out",
    "resid_pre_act": "hook_resid_pre",
    "resid_mid_act": "hook_resid_mid",
    "resid_post_act": "hook_resid_post",
}

def init_activation_store(model_names, num_layers):
    return {
        model_name: {
            l: {
                block_name: []
                for block_name in ACTIVATION_BLOCKS.keys()
            }
            for l in range(num_layers)
        }
        for model_name in model_names
    }

def append_activation(cache, store, model_name, num_layers) :
    for l in range(num_layers):
        for block_name, hook_name in ACTIVATION_BLOCKS.items():
            act = cache[f"blocks.{l}.{hook_name}"].detach().cpu()
            act = act.squeeze(0)  # [seq_len, d_model]

            store[model_name][l][block_name].append(act)

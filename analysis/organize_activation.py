import torch as th
import h5py
import numpy as np

ACTIVATION_BLOCKS = {
    "mlp_out_act": "hook_mlp_out",
    "attn_out_act": "hook_attn_out",
    "resid_pre_act": "hook_resid_pre",
}

def append_sample_index(
    index_group,
    sample_id,
    start,
    end,
    prompt_len,
    completion_len,
    total_len,
):
    append_1d_dataset(index_group, "sample_id", sample_id)
    append_1d_dataset(index_group, "start", start)
    append_1d_dataset(index_group, "end", end)
    append_1d_dataset(index_group, "prompt_len", prompt_len)
    append_1d_dataset(index_group, "completion_len", completion_len)
    append_1d_dataset(index_group, "total_len", total_len)

def append_to_h5_dataset(group, name, array, compression = "lzf"):
    array = np.asanyarray(array)

    if name not in group:
        maxshape = (None, array.shape[-1])

        group.create_dataset(
            name, 
            data = array,
            maxshape = maxshape,
            chunks = (min(1024, array.shape[0]), array.shape[1]),
            compression = compression
        )

        start = 0
        end = array.shape[0]
        return start, end
    
    dset = group[name]

    start = dset.shape[0]
    end = start + array.shape[0]

    dset.resize((end, array.shape[1]))
    dset[start:end] = array

    return start, end

def append_1d_dataset(group, name, value, dtype=np.int64):
    value = np.asarray([value], dtype=dtype)

    if name not in group:
        group.create_dataset(
            name,
            data=value,
            maxshape=(None,),
            chunks=True,
        )
        return

    dset = group[name]
    old_len = dset.shape[0]
    new_len = old_len + 1

    dset.resize((new_len,))
    dset[old_len:new_len] = value

def append_cache_to_single_h5(
    h5_path, 
    cache, 
    item, 
    model_name,
    interesting_layers,
    actication_dtype = np.float16
):
    sample_id = int(item["sample_id"])
    total_len = int(item["total_len"])

    with h5py.File(h5_path, "a") as f:
        model_group = f.require_group(model_name)

        # Save index for model
        index_group = model_group.require_group("index")

        # Write all actication : model/layer/block
        sample_start = None
        sample_end = None

        for l in interesting_layers:
            layer_group = model_group.require_group(f"layer_{l:02d}")

            for block_name, hook_name in ACTIVATION_BLOCKS.items():
                key = f"blocks.{l}.{hook_name}"

                act = cache[key].detach().cpu().squeeze(0)
                act_np = act.to(th.float16).numpy().astype(actication_dtype)

                start, end = append_to_h5_dataset(
                    group=layer_group,
                    name = block_name,
                    array=act_np
                )

                if sample_start is None: 
                    sample_start = start
                    sample_end = end
                
                del act, act_np

        append_sample_index(
            index_group=index_group,
            sample_id=sample_id,
            start=sample_start,
            end=sample_end,
            prompt_len=int(item["prompt_len"]),
            completion_len=int(item["completion_len"]),
            total_len=total_len,
        )

        f.flush()


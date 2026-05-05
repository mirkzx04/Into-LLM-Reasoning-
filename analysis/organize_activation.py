import h5py
import numpy as np
import torch as th

# Map user-facing activation names to TransformerLens hook names.
ACTIVATION_BLOCKS = {
    "mlp_out_act": "hook_mlp_out",
    "attn_out_act": "hook_attn_out",
    "resid_pre_act": "hook_resid_pre",
}


def append_1d_dataset(group, name, value, dtype=np.int64):
    # Append one scalar value to a resizable 1D dataset.
    value = np.asarray([value], dtype=dtype)

    if name not in group:
        group.create_dataset(
            name,
            data=value,
            maxshape=(None,),
            chunks=True,
        )
        return

    dataset = group[name]
    old_len = dataset.shape[0]
    new_len = old_len + 1
    dataset.resize((new_len,))
    dataset[old_len:new_len] = value


def append_sample_index(
    index_group,
    sample_id,
    start,
    end,
    prompt_len,
    completion_len,
    total_len,
):
    # Save the row span written for one sample inside the HDF5 file.
    append_1d_dataset(index_group, "sample_id", sample_id)
    append_1d_dataset(index_group, "start", start)
    append_1d_dataset(index_group, "end", end)
    append_1d_dataset(index_group, "prompt_len", prompt_len)
    append_1d_dataset(index_group, "completion_len", completion_len)
    append_1d_dataset(index_group, "total_len", total_len)


def append_to_h5_dataset(group, name, array, compression="lzf"):
    # Append a 2D activation matrix to a resizable HDF5 dataset.
    array = np.asarray(array)

    if name not in group:
        dataset = group.create_dataset(
            name,
            data=array,
            maxshape=(None, array.shape[-1]),
            chunks=(min(1024, array.shape[0]), array.shape[1]),
            compression=compression,
        )
        return 0, dataset.shape[0]

    dataset = group[name]
    start = dataset.shape[0]
    end = start + array.shape[0]
    dataset.resize((end, array.shape[1]))
    dataset[start:end] = array
    return start, end


def append_cache_to_single_h5(
    h5_path,
    cache,
    item,
    model_name,
    interesting_layers,
    activation_dtype=np.float16,
):
    # Append one sample worth of activations to the shared HDF5 file.
    sample_id = int(item["sample_id"])
    total_len = int(item["total_len"])

    with h5py.File(h5_path, "a") as h5_file:
        model_group = h5_file.require_group(model_name)
        index_group = model_group.require_group("index")

        # The same token span is shared by all blocks for the current sample.
        sample_start = None
        sample_end = None

        for layer in interesting_layers:
            layer_group = model_group.require_group(f"layer_{layer:02d}")

            for block_name, hook_name in ACTIVATION_BLOCKS.items():
                cache_key = f"blocks.{layer}.{hook_name}"
                activation = cache[cache_key].detach().cpu().squeeze(0)
                activation_np = activation.to(th.float16).numpy().astype(activation_dtype)

                start, end = append_to_h5_dataset(
                    group=layer_group,
                    name=block_name,
                    array=activation_np,
                )

                # Record the sample span only once because it is identical for all blocks.
                if sample_start is None:
                    sample_start = start
                    sample_end = end

                del activation
                del activation_np

        append_sample_index(
            index_group=index_group,
            sample_id=sample_id,
            start=sample_start,
            end=sample_end,
            prompt_len=int(item["prompt_len"]),
            completion_len=int(item["completion_len"]),
            total_len=total_len,
        )

        h5_file.flush()

import os 
import re
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import h5py
import numpy as np

ACTIVATION_DATASET_PATH = "activation_dataset"

RLVR_PATH = "rlvr_model_math"
SFT_PATH = "sftt_model_math"

DERIVED_RESIDUAL_COMPONENTS = {
    "attn_resid": ("resid_pre_act", "attn_out_act"),
    "mlp_resid": ("resid_pre_act", "attn_out_act", "mlp_out_act"),
}

def get_activation_dataset():
    from analysis.extract_activation import (
        extract_activation,
        get_activation_dataset_paths,
        load_existing_activation_dataset,
    )

    max_new_tokens = 2000
    generator_name = "qwen25_1.5b_rlvr"
    ood_dataset_name = "ood_eval_dataset"

    dataset_paths = get_activation_dataset_paths(
        save_path=ACTIVATION_DATASET_PATH,
        max_new_tokens=max_new_tokens,
        generator_name=generator_name,
        ood_dataset_name=ood_dataset_name,
    )

    if os.path.exists(dataset_paths["h5_path"]):
        return load_existing_activation_dataset(
            h5_path=dataset_paths["h5_path"],
            metadata_path=dataset_paths["metadata_path"],
        )

    from models.model import get_hf_model, get_tokenizer
    from MATH_logic.dataset_utils.dataset_splitting import build_ood_eval_dataset

    

    gen_model = get_hf_model(RLVR_PATH)
    gen_tokenizer = get_tokenizer(RLVR_PATH)
    gen_dataset = build_ood_eval_dataset(
        tokenizer=gen_tokenizer,
        mode="rlvr"
    )

    model_desc = [(None, "base"), (SFT_PATH, "sftt"), (RLVR_PATH, "rlvr")]
    
    return extract_activation(
        max_new_tokens=max_new_tokens,
        batch_size=40,
        gen_model=gen_model,
        gen_tokenizer=gen_tokenizer,
        gen_dataset=gen_dataset,
        model_desc=model_desc,
        generator_name=generator_name,
        ood_dataset_name=ood_dataset_name,
        save_path=ACTIVATION_DATASET_PATH
    )

def get_requested_act_modules(act_modules):
    """
    Normalizza la richiesta utente in una lista di chiavi output.
    Supporta:
    - base: resid_pre_act, attn_out_act, mlp_out_act
    - derivate: attn_resid, mlp_resid
    - shortcut: resids -> resid_pre, attn_resid, mlp_resid
    """
    if act_modules is None:
        return ["resid_pre_act", "attn_out_act", "mlp_out_act"]

    if isinstance(act_modules, str):
        act_modules = [act_modules]

    out = []
    for act in act_modules:
        if act == "resids":
            out.extend(["resid_pre", "attn_resid", "mlp_resid"])
        else:
            out.append(act)

    # dedup preservando l'ordine
    seen = set()
    dedup_out = []
    for act in out:
        if act not in seen:
            dedup_out.append(act)
            seen.add(act)

    return dedup_out

def extract_model_groups(model_names, h5_file): 
    if model_names is None:
        model_groups = list(h5_file.keys())
    else: 
        if isinstance(model_names, str):
            model_groups = [model_names]
        elif isinstance(model_names, list):
            model_groups = model_names
        else:
            raise TypeError("model_names must be None, str or list[str]")

    return model_groups

def extract_layer_groups(model, layers):
    if layers is None:
        layer_groups = [k for k in model.keys() if k.startswith("layer")]
        layer_groups = sorted(layer_groups, key=layer_group_sort_key)
    else: 
        if isinstance(layers, str):
            layer_groups = [layers]
        elif isinstance(layers, list):
            layer_groups = layers
        else:
            raise TypeError("layers must be None, str or list[str]")
    
    return layer_groups


def layer_group_sort_key(layer_name):
    match = re.search(r"(\d+)$", layer_name)
    if match is None:
        return (1, layer_name)

    return (0, int(match.group(1)))


def format_layer_group_label(layer_name):
    match = re.search(r"(\d+)$", layer_name)
    if match is None:
        return layer_name

    return str(int(match.group(1)))


def resolve_saved_layer_labels(h5_path, model_name=None, layers=None):
    """
    Resolve display labels from saved activation artifacts without loading tensors.
    Reads only the HDF5 group structure.
    """
    with h5py.File(h5_path, "r") as f:
        model_groups = extract_model_groups(model_names=model_name, h5_file=f)

        if not model_groups:
            raise ValueError(f"No model groups found in activation dataset: {h5_path}")

        model_group = f[model_groups[0]]
        layer_groups = extract_layer_groups(model=model_group, layers=layers)

    return [format_layer_group_label(layer_name) for layer_name in layer_groups]

def extract_act_module_groups(l_group, act_modules):
    # Mantengo il metodo, ma ora ritorna nomi logici richiesti.
    if act_modules is None:
        act_module_groups = list(l_group.keys())
    else: 
        if isinstance(act_modules, str):
            act_module_groups = [act_modules]
        elif isinstance(act_modules, list):
            act_module_groups = act_modules
        else:
            raise TypeError("act_modules must be None, str or list[str]")

    return act_module_groups

def get_batch_index(index_group, start_idx, batch_size):
    sample_ids = index_group["sample_id"][start_idx:start_idx + batch_size]
    starts = index_group["start"][start_idx:start_idx + batch_size]
    ends = index_group["end"][start_idx:start_idx + batch_size]
    prompt_lens = index_group["prompt_len"][start_idx:start_idx + batch_size]
    completion_lens = index_group["completion_len"][start_idx:start_idx + batch_size]
    total_lens = index_group["total_len"][start_idx:start_idx + batch_size]
    return sample_ids, starts, ends, prompt_lens, completion_lens, total_lens


def get_selected_sample_rows(n_samples, max_sample, sample_seed):
    selected_rows = np.arange(n_samples)

    if sample_seed is not None:
        rng = np.random.default_rng(sample_seed)
        selected_rows = rng.permutation(selected_rows)

    if max_sample is not None:
        selected_rows = selected_rows[: min(n_samples, max_sample)]

    return selected_rows


def get_index_rows(index_group, row_indices):
    row_indices = np.asarray(row_indices)
    sort_order = np.argsort(row_indices)
    sorted_row_indices = row_indices[sort_order]
    restore_order = np.argsort(sort_order)

    sample_ids = index_group["sample_id"][sorted_row_indices][restore_order]
    starts = index_group["start"][sorted_row_indices][restore_order]
    ends = index_group["end"][sorted_row_indices][restore_order]
    prompt_lens = index_group["prompt_len"][sorted_row_indices][restore_order]
    completion_lens = index_group["completion_len"][sorted_row_indices][restore_order]
    total_lens = index_group["total_len"][sorted_row_indices][restore_order]
    return sample_ids, starts, ends, prompt_lens, completion_lens, total_lens

def slice_samples_from_act(act_dataset, starts, ends):
    return [act_dataset[s:e] for s, e in zip(starts, ends)]

def build_samples_for_act(layer_group, act_name, starts, ends):
    """
    Costruisce i sample per un singolo layer.
    Supporta sia attivazioni salvate direttamente sia residuali derivati.
    I residuali derivati sono cumulativi:
    - attn_resid = resid_pre_act + attn_out_act
    - mlp_resid = resid_pre_act + attn_out_act + mlp_out_act
    """
    if act_name == "resid_pre":
        return slice_samples_from_act(
            act_dataset=layer_group["resid_pre_act"],
            starts=starts,
            ends=ends,
        )

    if act_name in DERIVED_RESIDUAL_COMPONENTS:
        missing_components = [
            component
            for component in DERIVED_RESIDUAL_COMPONENTS[act_name]
            if component not in layer_group
        ]
        if missing_components:
            raise KeyError(
                f"Cannot build cumulative {act_name}; missing components: {missing_components}"
            )

        component_samples = [
            slice_samples_from_act(
                act_dataset=layer_group[component],
                starts=starts,
                ends=ends,
            )
            for component in DERIVED_RESIDUAL_COMPONENTS[act_name]
        ]

        return [
            sum(sample_parts[1:], sample_parts[0].copy())
            for sample_parts in zip(*component_samples)
        ]

    # Caso base: attivazione già salvata in HDF5
    return slice_samples_from_act(
        act_dataset=layer_group[act_name],
        starts=starts,
        ends=ends,
    )

def stack_layers_for_samples(layer_groups, model_group, act_name, starts, ends):
    per_layer_samples = []

    for layer_name in layer_groups:
        layer_group = model_group[layer_name]
        layer_samples = build_samples_for_act(
            layer_group=layer_group,
            act_name=act_name,
            starts=starts,
            ends=ends,
        )
        per_layer_samples.append(layer_samples)

    # per_layer_samples: list[L] of list[B] of [seq_len_i, d_model]
    batch_out = []
    batch_size = len(starts)

    for b in range(batch_size):
        sample_layers = [per_layer_samples[l][b] for l in range(len(per_layer_samples))]
        batch_out.append(np.stack(sample_layers, axis=0))   # [L, seq_len_i, d_model]

    return batch_out

def load_sample_batch(
    batch_size,
    model_names,
    act_modules,
    layers,
    h5_path,
    max_sample,
    sample_seed=42,
):
    activation_out = {}

    requested_act_modules = get_requested_act_modules(act_modules)

    with h5py.File(h5_path, "r") as f:
        model_groups = extract_model_groups(model_names=model_names, h5_file=f)
        selected_rows = None

        for model_name in model_groups:
            model_group = f[model_name]

            activation_out[model_name] = {
                "prompt_len": {},
                "completion_len": {},
                "total_len": {},
                **{act_name: {} for act_name in requested_act_modules}
            }

            index_group = model_group["index"]
            n_samples = len(index_group["sample_id"])

            if selected_rows is None:
                selected_rows = get_selected_sample_rows(
                    n_samples=n_samples,
                    max_sample=max_sample,
                    sample_seed=sample_seed,
                )

            layer_groups = extract_layer_groups(model=model_group, layers=layers)

            for i in range(0, len(selected_rows), batch_size):
                batch_rows = selected_rows[i:i + batch_size]
                sample_ids, starts, ends, prompt_lens, completion_lens, total_lens = get_index_rows(
                    index_group=index_group,
                    row_indices=batch_rows,
                )

                for sid, p_len, c_len, t_len in zip(sample_ids, prompt_lens, completion_lens, total_lens):
                    sid = int(sid)
                    activation_out[model_name]["prompt_len"][sid] = int(p_len)
                    activation_out[model_name]["completion_len"][sid] = int(c_len)
                    activation_out[model_name]["total_len"][sid] = int(t_len)

                for act_name in requested_act_modules:
                    batch_samples = stack_layers_for_samples(
                        layer_groups=layer_groups,
                        act_name=act_name,
                        starts=starts,
                        ends=ends,
                        model_group=model_group
                    )

                    for sid, sample_arr in zip(sample_ids, batch_samples):
                        activation_out[model_name][act_name][int(sid)] = sample_arr

    return activation_out, layer_groups

import os 
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import h5py

from analysis.extract_activation import extract_activation, get_model, get_tokenizer
from MATH_logic.dataset_utils.dataset_splitting import build_ood_eval_dataset

ACTIVATION_DATASET_PATH = "activation_dataset"

RLVR_PATH = "rlvr_model_math"
SFT_PATH = "sftt_model_math"

VALID_MODULES = {
    "mlp_output_act",
    "attn_out_act",
    "resid_pre_act"
}

def get_activation_dataset():
    gen_model = get_model(RLVR_PATH)
    gen_tokenizer = get_tokenizer(RLVR_PATH)
    gen_dataset = build_ood_eval_dataset(
        tokenizer=gen_tokenizer,
        mode="rlvr"
    )

    model_desc = [(None, "base"), (SFT_PATH, "sftt"), (RLVR_PATH, "rlvr")]
    
    return extract_activation(
        max_new_tokens=2000,
        batch_size= 40,
        gen_model= gen_model,
        gen_tokenizer= gen_tokenizer,
        gen_dataset= gen_dataset,
        model_desc= model_desc,
        generator_name="qwen25_1.5b_rlvr",
        ood_dataset_name="ood_eval_dataset",
        save_path=ACTIVATION_DATASET_PATH
    )

def extract_model_groups(model_names, h5_file): 
    if model_names is None:
        model_groups = list(h5_file.keys())
    else : 
        if type(model_names) == str:
            model_groups = [h5_file[model_names]]
        elif type(model_names) == list :
            model_groups = [h5_file[m_name] for m_name in model_names]

    return model_groups

def extract_layer_groups(model, layers):
    if layers is None:
        layer_groups = [k for k in model.keys() if k.startswith("layer")]
    else : 
        if type(layers) == str :
            layer_groups = [model[layers]]
        elif type(layers) == list:
            layer_groups = [model[l] for l in layers]
    
    return layer_groups

def extract_act_module_groups(l_group, act_modules):
    if act_modules is None:
        act_module_groups = list(l_group.keys())
    else : 
        if type(act_modules) == str :
            act_module_groups = [l_group[act_modules]]
        elif type(act_modules) == list:
            act_module_groups = [l_group[act] for act in act_modules]

    return act_module_groups

import h5py
import numpy as np

def get_batch_index(index_group, start_idx, batch_size):
    sample_ids = index_group["sample_id"][start_idx:start_idx + batch_size]
    starts = index_group["start"][start_idx:start_idx + batch_size]
    ends = index_group["end"][start_idx:start_idx + batch_size]
    sample_ids = index_group["sample_id"][start_idx:start_idx + batch_size]
    starts = index_group["start"][start_idx:start_idx + batch_size]
    ends = index_group["end"][start_idx:start_idx + batch_size]
    prompt_lens = index_group["prompt_len"][start_idx:start_idx + batch_size]
    completion_lens = index_group["completion_len"][start_idx:start_idx + batch_size]
    total_lens = index_group["total_len"][start_idx:start_idx + batch_size]
    return sample_ids, starts, ends, prompt_lens, completion_lens, total_lens


def slice_samples_from_act(act_dataset, starts, ends):
    return [act_dataset[s:e] for s, e in zip(starts, ends)]


def stack_layers_for_samples(layer_groups, act_name, starts, ends):
    per_layer_samples = []

    for l_group in layer_groups:
        act_dataset = l_group[act_name]
        layer_samples = slice_samples_from_act(
            act_dataset=act_dataset,
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
):
    activation_out = {}

    with h5py.File(h5_path, "r") as f:
        model_groups = extract_model_groups(model_names=model_names, h5_file=f)

        for m_group in model_groups:
            model_name = m_group.name.split("/")[-1]
            activation_out[model_name] = {
                "prompt_len": {},
                "completion_len": {},
                "total_len": {},
                **{act_name: {} for act_name in act_modules}
            }

            index_group = m_group["index"]
            n_samples = len(index_group["sample_id"])

            layer_groups = extract_layer_groups(model=m_group, layers=layers)

            for i in range(0, n_samples, batch_size):
                sample_ids, starts, ends, prompt_lens, completion_lens, total_lens = get_batch_index(
                    index_group=index_group,
                    start_idx=i,
                    batch_size=batch_size,
                )

                for sid, p_len, c_len, t_len in zip(sample_ids, prompt_lens, completion_lens, total_lens):
                    sid = int(sid)
                    activation_out[model_name]["prompt_len"][sid] = int(p_len)
                    activation_out[model_name]["completion_len"][sid] = int(c_len)
                    activation_out[model_name]["total_len"][sid] = int(t_len)

                for act_name in act_modules:
                    batch_samples = stack_layers_for_samples(
                        layer_groups=layer_groups,
                        act_name=act_name,
                        starts=starts,
                        ends=ends,
                    )

                    for sid, sample_arr in zip(sample_ids, batch_samples):
                        activation_out[model_name][act_name][int(sid)] = sample_arr

    return activation_out
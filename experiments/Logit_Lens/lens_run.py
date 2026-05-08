import os
import sys
from itertools import combinations
from typing import Optional

import torch as th
import torch.nn as nn

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from experiments.act_dataset_utils import (
    RLVR_PATH,
    SFT_PATH,
    get_activation_dataset,
    load_sample_batch,
)
from experiments.Logit_Lens.get_lens import get_lens
from experiments.Logit_Lens.lens_utils import lens_view


LensMap = dict[str, nn.Module]
LensBank = dict[str, nn.Module | LensMap]


def create_model_comparison(model_names: list[str]) -> list[tuple[str, str]]:
    # Build all model pairs used by pairwise metrics.
    model_pairs: list[tuple[str, str]] = list(combinations(model_names, 2))
    return model_pairs


def build_model_path_map() -> dict[str, Optional[str]]:
    # "base" uses the default HF model.
    # The other entries point to local checkpoints.
    model_paths: dict[str, Optional[str]] = {
        "base": None,
        "sftt": SFT_PATH,
        "rlvr": RLVR_PATH,
    }
    return model_paths


def build_bank_lens(
    model_names: list[str],
    device: str,
    shared_lens_source: str,
) -> LensBank:
    # bank_lens["native"][model_name]:
    # each model uses its own lens.
    #
    # bank_lens["shared"]:
    # all models reuse the lens chosen by shared_lens_source.
    model_paths: dict[str, Optional[str]] = build_model_path_map()
    native_lenses: LensMap = {}

    for model_name in model_names:
        if model_name not in model_paths:
            raise ValueError(f"Unknown model name: {model_name}")

        model_path: Optional[str] = model_paths[model_name]
        lens: nn.Module = get_lens(model_path, device)

        native_lenses[model_name] = lens

    if shared_lens_source not in native_lenses:
        raise ValueError(f"Unknown shared lens source: {shared_lens_source}")

    bank_lens: LensBank = {
        "native": native_lenses,
        "shared": native_lenses[shared_lens_source],
    }
    return bank_lens


def save_lens_out(
    lens_out: dict,
    lens_modes: list[str],
    shared_lens_source: str,
) -> str:
    # Save lens_out to a local .pt file.
    output_dir: str = os.path.join(project_root, "experiments", "Logit_Lens", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    lens_mode_tag: str = "_".join(lens_modes)
    output_name: str = f"lens_out_{lens_mode_tag}_shared-{shared_lens_source}.pt"
    output_path: str = os.path.join(output_dir, output_name)

    th.save(lens_out, output_path)
    return output_path


def main() -> None:
    device: str = "cuda" if th.cuda.is_available() else "cpu"

    act_dataset: dict[str, str] = get_activation_dataset()
    h5_path: str = act_dataset["h5_path"]

    act_modules: list[str] = ["resid_pre", "attn_resid", "mlp_resid"]
    normalized_positions: list[float] = [0.1]
    layers: Optional[list[str]] = None

    activation_out, _layer_ids = load_sample_batch(
        batch_size=10,
        model_names=None,
        act_modules=act_modules,
        layers=layers,
        h5_path=h5_path,
        max_sample=100,
    )

    model_names: list[str] = list(activation_out.keys())
    model_comparison: list[tuple[str, str]] = create_model_comparison(model_names)
    lens_modes: list[str] = ["shared"]
    shared_lens_source: str = "base"
    bank_lens: LensBank = build_bank_lens(
        model_names=model_names,
        device=device,
        shared_lens_source=shared_lens_source,
    )

    lens_out = lens_view(
        views=normalized_positions,
        act_names=act_modules,
        lens_modes=lens_modes,
        model_names=model_names,
        activation_out=activation_out,
        lens_bank=bank_lens,
        model_comparison=model_comparison,
        device=device,
    )
    output_path: str = save_lens_out(
        lens_out=lens_out,
        lens_modes=lens_modes,
        shared_lens_source=shared_lens_source,
    )

    print("Available lens modes:", lens_modes)
    print("Shared lens source:", shared_lens_source)
    print("Computed positions:", list(lens_out.keys()))
    print("Saved lens_out to:", output_path)


if __name__ == "__main__":
    main()

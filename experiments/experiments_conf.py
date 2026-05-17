import os
from dataclasses import dataclass, field
from typing import Any, Mapping


EXPERIMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EXPERIMENTS_DIR, ".."))
CKA_IMG_DIR = os.path.join(EXPERIMENTS_DIR, "CKA", "cka_img")
DEFAULT_CKA_ACTIVATION_H5_PATH = os.path.join(
    PROJECT_ROOT,
    "activation_dataset",
    "qwen25_1.5b_rlvr_ood_eval_dataset_max_2000_acts.h5",
)


@dataclass(frozen=True)
class LensConfig:
    """Configuration for Logit Lens runs."""

    positions: tuple = ("last_input_token", 0.1, 0.5, 0.9, 0.95)
    act_modules: tuple = ("resid_pre", "attn_resid", "mlp_resid")
    report_act_modules: tuple = ("attn_resid", "mlp_resid")
    lens_modes: tuple = ("native",)
    shared_lens_source: str = "rlvr"
    layers: object = None
    batch_size: int = 10
    max_sample: int = 500
    sample_selection_seed: int = 42
    model_names = None


@dataclass(frozen=True)
class PatchConfig:
    """Configuration for activation patching runs."""

    positions: tuple = ("last_input_token", 0.1, 0.5, 0.9, 0.95)
    patch_modules: tuple = ("attn_out", "mlp_out", "resid_mid", "resid_post")
    recivient_name: str = "base"
    donor_name: str = "rlvr"
    layers: object = None
    batch_size: int = 10
    max_sample: int = 500
    sample_selection_seed: int = 42
    model_names = ["rlvr", "base"]


@dataclass(frozen=True)
class CKAConfig:
    """Shared sampling configuration for CKA heatmaps."""

    layers: object = None
    batch_size: int = 5
    max_sample: int = 1000
    sample_seed: int = 42
    h5_path: str = DEFAULT_CKA_ACTIVATION_H5_PATH


@dataclass(frozen=True)
class CKAHeatmapConfig:
    """Configuration for a single CKA heatmap."""

    act_module: str
    slicing_mode: str
    output_path: str
    title: str
    slicing_kwargs: Mapping[str, Any] = field(default_factory=dict)


HEATMAP_CONFIGS = (
    CKAHeatmapConfig(
        act_module="attn_out_act",
        slicing_mode="predictive_completion_mean",
        output_path=os.path.join(CKA_IMG_DIR, "act", "attn_out_pred_comp_mean.png"),
        title="attn_out_act CKA - predictive_completion_mean",
    ),
    CKAHeatmapConfig(
        act_module="attn_out_act",
        slicing_mode="position_normalized_completion",
        slicing_kwargs={"normalized_pos": 0.25},
        output_path=os.path.join(
            CKA_IMG_DIR,
            "act",
            "attn_out_position_normalized_completion_025.png",
        ),
        title="attn_out_act CKA - position_normalized_completion=0.25",
    ),
    CKAHeatmapConfig(
        act_module="mlp_out_act",
        slicing_mode="last_input_token",
        output_path=os.path.join(CKA_IMG_DIR, "mlp", "mlp_out_last_inp_tok.png"),
        title="mlp_out_act CKA - last_input_token",
    ),
    CKAHeatmapConfig(
        act_module="mlp_out_act",
        slicing_mode="predictive_completion_mean",
        output_path=os.path.join(CKA_IMG_DIR, "mlp", "mlp_out_pred_comp_mean.png"),
        title="mlp_out_act CKA - predictive_completion_mean",
    ),
    CKAHeatmapConfig(
        act_module="mlp_out_act",
        slicing_mode="position_normalized_completion",
        slicing_kwargs={"normalized_pos": 0.25},
        output_path=os.path.join(CKA_IMG_DIR, "mlp", "mlp_norm_pos_025.png"),
        title="mlp_out_act CKA - position_normalized_completion=0.25",
    ),
)

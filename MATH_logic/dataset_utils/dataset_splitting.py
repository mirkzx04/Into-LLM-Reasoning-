
"""
Pipelines for loading, filtering, formatting, and splitting mathematical 
reasoning datasets (GSM8K, MATH, MetaMath) for model training.
"""
import os 
import sys
import re
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from datasets import load_dataset, concatenate_datasets

from MATH_logic.dataset_utils.dataset_formatting import (format_sft_example, 
format_rlvr_example, add_prompt_ids)

# Configuration: RNG seed, split sizes, and problem filtering criteria
SEED = 42
TEST_SIZE = 0.1

NUMINA_SAMPLES = 80_000

TRAIN_LEVELS = ["Level 1", "Level 2", "Level 3", "Level 4"]
TRAIN_TYPES = ["Intermediate Algebra", "Number Theory", "Prealgebra", "Precalculus"]

OOD_LEVELS = ["Level 5"]
OOD_TYPES = ["Algebra", "Counting & Probability", "Geometry"]

# Load raw datasets and standardize column names
numina_set = load_dataset("AI-MO/NuminaMath-CoT")
gsm8k_set = (
    load_dataset("openai/gsm8k", "main")
    .rename_columns({"question" : "problem", "answer" : "solution"})
)
math_set = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default")
metamath_train = (
    load_dataset("meta-math/MetaMathQA", split="train")
    .remove_columns({"query" : "problem", "response" : "solution"})
)

def normalize_txt(x) :
    """Normalizes text by removing extra whitespaces for exact string matching."""
    return re.sub(r"\s+", " ", str(x)).strip()

math_train_rows = math_set["train"]

# Precompute a lookup table to quickly access MATH examples metadata
MATH_LOOKUP = {
    normalize_txt(row["problem"]): {
        "level": row["level"],
        "type": row["type"],
    }
    for row in math_train_rows
}

def filter_math(example, levels, types):
    """Filters MATH dataset examples based on specified difficulty levels and types."""
    return example["level"] in levels and example["type"] in types

def filter_metamath(example, levels, types) : 
    """Filters MetaMath examples by retrieving their original MATH metadata."""
    original = normalize_txt(example.get("original_question", ""))
    meta = MATH_LOOKUP.get(original)

    if meta is None:
        return False

    return meta["level"] in levels and meta["type"] in types

def split_train_val(dataset) : 
    """Splits a pre-shuffled dataset into training and validation sets."""
    split = dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
    return split["train"], split["test"]

def sample_dataset(dataset, n_samples):
    """Randomly subsamples a specific number of instances from a dataset."""
    n_samples = min(n_samples, len(dataset))

    return (
        dataset
        .shuffle(seed=SEED)
        .select(range(n_samples))
    )

def format_dataset(dataset, tokenizer, dataset_name, mode):
    """Applies specific prompt formatting to the dataset based on the training mode (SFT or RLVR)."""
    
    if "prompt_id" not in dataset.column_names : 
        dataset = add_prompt_ids(dataset, seed = SEED)

    if mode == "sft" :
        return dataset.map(
            lambda example: format_sft_example(
                example,
                tokenizer,
                dataset_name=dataset_name, 
                prompt_id=example["prompt_id"]
            ),
            remove_columns=dataset.column_names
        )
    if mode == "rlvr" :  
        return dataset.map(
            lambda example: format_rlvr_example(
                example,
                dataset_name=dataset_name, 
                prompt_id=example["prompt_id"]
            ),
            remove_columns=dataset.column_names
        )
    
    raise ValueError(f"Unknown mode: {mode}")

def build_mixed_dataset(parts, tokenizer, training) :
    """Samples, formats, mixes, and splits multiple dataset parts into a unified train/val set."""
    formatted_sets = []
    for part in parts : 
        sampled = sample_dataset(part["dataset"], part["n_samples"])

        formatted = format_dataset(
            sampled,
            tokenizer,
            dataset_name=part["dataset_name"],
            training=training
        )

        formatted_sets.append(formatted)

    mixed = concatenate_datasets(formatted_sets)
    mixed = mixed.shuffle(seed=SEED)

    split = split_train_val(mixed)

    return split["train"], split["test"]

def build_train_val_dataset(tokenizer, training):
    """Builds the main training mix combining filtered subsets of GSM8K, MATH, and MetaMath."""
    gsm_train = gsm8k_set["train"]
    math_train = math_set["train"].filter(
        lambda x : filter_math(x, TRAIN_LEVELS, TRAIN_TYPES)
    )
    metamath_filtered = metamath_train.filter(
        lambda x : filter_metamath(x, TRAIN_LEVELS, TRAIN_TYPES)
    )

    parts = [
        {
            "dataset": gsm_train,
            "dataset_name": "gsm8k",
            "n_samples": None,
        },
        {
            "dataset": math_train,
            "dataset_name": "math",
            "n_samples": None,
        },
        {
            "dataset": metamath_filtered,
            "dataset_name": "metamath",
            "n_samples": 20_000,
        },
    ]

    return build_mixed_dataset(parts=parts, tokenizer=tokenizer, training=training)

def build_ood_eval_dataset(tokenizer, mode="rlvr") :
    """Constructs an Out-Of-Distribution evaluation dataset for testing generalization."""
    gsm_tst = gsm8k_set["test"]

    math_ood = math_set["train"].filter(
        lambda x : filter_math(x, OOD_LEVELS, OOD_TYPES)
    )

    parts = [
        {
            "dataset": gsm_tst,
            "dataset_name": "gsm8k",
        },
        {
            "dataset": math_ood,
            "dataset_name": "math",
        },
    ]

    formatted_sets = []

    for part in parts:
        ds = format_dataset(
            part["dataset"],
            tokenizer,
            dataset_name=part["dataset_name"],
            mode=mode,
        )
        formatted_sets.append(ds)

    return concatenate_datasets(formatted_sets).shuffle(seed=SEED)
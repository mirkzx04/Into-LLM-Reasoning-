import os
import sys
import re
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

import shutil
import hashlib

from datasets import load_dataset, concatenate_datasets, load_from_disk

from MATH_logic.dataset_utils.dataset_formatting import (
    format_sft_example,
    format_rlvr_example,
    add_prompt_ids,
    is_extractable_example,
)

DATA_DIR = os.path.join(project_root, "data")
OOD_SAVE_DIR = os.path.join(DATA_DIR, "ood_eval_dataset")

SEED = 42
TEST_SIZE = 0.01

NUMINA_SAMPLES = 80_000

TRAIN_LEVELS = ["Level 1", "Level 2", "Level 3", "Level 4"]
TRAIN_TYPES = ["Intermediate Algebra", "Number Theory", "Prealgebra", "Precalculus"]

OOD_LEVELS = ["Level 5"]
OOD_TYPES = ["Algebra", "Counting & Probability", "Geometry"]

_DATASETS_CACHE = None
_MATH_LOOKUP_CACHE = None

def load_dataset_lazily():
    """
    Load dataset and build the lookup table
    """

    global _DATASETS_CACHE, _MATH_LOOKUP_CACHE

    if _DATASETS_CACHE is not None:
        return

    print("[Lazy Load] Caricamento dei dataset in corso (verrà eseguito una volta sola)...")
    
    _DATASETS_CACHE = {}
    
    _DATASETS_CACHE["numina"] = load_dataset("AI-MO/NuminaMath-CoT")
    
    _DATASETS_CACHE["gsm8k"] = (
        load_dataset("openai/gsm8k", "main")
        .rename_columns({"question": "problem", "answer": "solution"})
    )
    
    _DATASETS_CACHE["math"] = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default")
    
    _DATASETS_CACHE["dapo"] = load_dataset(
        "open-r1/DAPO-Math-17k-Processed",
        "en",
        split="train",
    )

    print("[Lazy Load] Costruzione del MATH_LOOKUP...")
    math_train_rows = _DATASETS_CACHE["math"]["train"]
    _MATH_LOOKUP_CACHE = {
        normalize_for_lookup(row["problem"]): {
            "level": row["level"],
            "type": row["type"],
        }
        for row in math_train_rows
    }
    print("[Lazy Load] Caricamento completato.")

def canonical_problem_text(text):
    """
    Stronger normalization for exact/near-exact duplicate detection.
    This is not semantic deduplication, but catches most textual leakage.
    """
    text = str(text)
    text = text.replace("\r\n", "\n")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[`*_]+", "", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text


def problem_fingerprint(problem):
    canonical = canonical_problem_text(problem)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def add_leakage_metadata(dataset, source_name):
    """
    Adds source + fingerprint columns to raw datasets before formatting.
    Requires a 'problem' column.
    """
    return dataset.map(
        lambda ex: {
            "leak_source": source_name,
            "problem_fingerprint": problem_fingerprint(ex["problem"]),
            "problem_canonical": canonical_problem_text(ex["problem"]),
        },
        desc=f"Adding leakage metadata [{source_name}]",
    )


def collect_fingerprint_map(dataset):
    """
    Returns:
        fingerprint -> list of examples metadata
    """
    out = {}

    for ex in dataset:
        fp = ex["problem_fingerprint"]
        out.setdefault(fp, []).append({
            "source": ex.get("leak_source", "unknown"),
            "problem": ex.get("problem", ""),
        })

    return out

def collect_train_fingerprints(train_raw_parts):
    train_all = concatenate_datasets(train_raw_parts)
    return set(train_all["problem_fingerprint"])


def remove_train_overlap_from_ood(dataset, train_fingerprints, source_name):
    """
    Removes from an OOD candidate dataset all rows whose normalized problem
    already appears in the RLVR train pool.
    """
    dataset = add_leakage_metadata(dataset, source_name)

    before = len(dataset)

    dataset = dataset.filter(
        lambda ex: ex["problem_fingerprint"] not in train_fingerprints,
        desc=f"Removing train/OOD leakage [{source_name}]",
    )

    after = len(dataset)

    print(
        f"[Leakage removal] source={source_name} "
        f"kept={after}/{before} removed={before - after}"
    )

    return dataset

def assert_no_train_ood_overlap(train_raw_parts, ood_raw_parts, max_print=20):
    """
    Checks exact normalized problem leakage between train and OOD.

    train_raw_parts and ood_raw_parts must already have:
    - leak_source
    - problem_fingerprint
    - problem_canonical
    """
    train_all = concatenate_datasets(train_raw_parts)
    ood_all = concatenate_datasets(ood_raw_parts)

    train_map = collect_fingerprint_map(train_all)
    ood_map = collect_fingerprint_map(ood_all)

    train_fps = set(train_map.keys())
    ood_fps = set(ood_map.keys())

    overlap = sorted(train_fps & ood_fps)

    print(f"[Leakage check] train_rows={len(train_all)}")
    print(f"[Leakage check] ood_rows={len(ood_all)}")
    print(f"[Leakage check] exact_problem_overlap={len(overlap)}")

    if overlap:
        print("[Leakage check] Examples:")

        for fp in overlap[:max_print]:
            train_ex = train_map[fp][0]
            ood_ex = ood_map[fp][0]

            print("=" * 100)
            print(f"fingerprint={fp}")
            print(f"TRAIN source={train_ex['source']}")
            print(train_ex["problem"][:1000])
            print("-" * 100)
            print(f"OOD source={ood_ex['source']}")
            print(ood_ex["problem"][:1000])

        raise ValueError(
            f"Train/OOD leakage detected: {len(overlap)} exact normalized problem overlaps."
        )

    return {
        "train_rows": len(train_all),
        "ood_rows": len(ood_all),
        "exact_problem_overlap": 0,
    }

def normalize_txt(x):
    return re.sub(r"\s+", " ", str(x)).strip()

def add_source_column(dataset, source_name):
    """
    Adds source metadata after formatting.

    This keeps the final OOD dataset analyzable by source:
    - gsm8k_test
    - math_ood
    - dapo_ood
    """
    return dataset.add_column(
        "ood_source",
        [source_name] * len(dataset),
    )


def save_dataset_to_local(dataset, save_dir, overwrite=True, save_jsonl=True):
    """
    Saves the full HuggingFace Dataset locally.

    save_to_disk() is the canonical format.
    jsonl is optional, useful for manual inspection.
    """
    if overwrite and os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    dataset.save_to_disk(save_dir)

    if save_jsonl:
        jsonl_path = save_dir + ".jsonl"
        dataset.to_json(jsonl_path, force_ascii=False)

    metadata = {
        "num_rows": len(dataset),
        "columns": dataset.column_names,
        "save_dir": save_dir,
        "jsonl_path": save_dir + ".jsonl" if save_jsonl else None,
    }

    metadata_path = save_dir + "_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[OOD save] saved dataset to: {save_dir}")
    print(f"[OOD save] rows={len(dataset)} columns={dataset.column_names}")

    return dataset

def normalize_for_lookup(x):
    x = str(x)
    x = x.replace("\r\n", "\n")
    x = re.sub(r"\s+", " ", x).strip()
    x = x.replace("\\left", "").replace("\\right", "")
    return x


def split_train_val(dataset):
    split = dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
    return split["train"], split["test"]


def sample_dataset(dataset, n_samples=None):
    dataset = dataset.shuffle(seed=SEED)

    if n_samples is None:
        return dataset

    n_samples = min(int(n_samples), len(dataset))
    return dataset.select(range(n_samples))

    math_train_rows = math_set["train"]

    MATH_LOOKUP = {
        normalize_for_lookup(row["problem"]): {
            "level": row["level"],
            "type": row["type"],
        }
        for row in math_train_rows
    }


def filter_math(example, levels, types):
    return example["level"] in levels and example["type"] in types


def get_math_meta_from_problem(problem):
    load_dataset_lazily()
    return _MATH_LOOKUP_CACHE.get(normalize_for_lookup(problem))

def get_dapo_prompt_content(example):
    prompt = example.get("prompt", "")

    if isinstance(prompt, list):
        if len(prompt) == 0:
            return ""

        first = prompt[0]

        if isinstance(first, dict):
            return str(first.get("content", "")).strip()

        return str(first).strip()

    return str(prompt).strip()


def clean_dapo_problem(text):
    """
    DAPO prompt format usually contains an instruction prefix and an answer-format suffix.
    This function keeps only the actual math problem.
    """
    text = str(text).strip().replace("\r\n", "\n")

    # Remove leading instruction block.
    # Example:
    # "Solve the following math problem step by step. ...\n\n<actual problem>"
    parts = text.split("\n\n")
    if len(parts) >= 2:
        first = parts[0].lower()
        if "solve" in first and "problem" in first:
            text = "\n\n".join(parts[1:]).strip()

    # Remove trailing DAPO answer-format reminder.
    text = re.sub(
        r"\n\s*Remember to put your answer.*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    # Remove possible final answer-format instruction if present.
    text = re.sub(
        r"\n\s*The last line of your response should be.*$",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    return text


def get_dapo_ground_truth(example):
    reward_model = example.get("reward_model", {})

    if isinstance(reward_model, str):
        try:
            reward_model = json.loads(reward_model)
        except Exception:
            reward_model = {}

    if isinstance(reward_model, dict):
        gt = reward_model.get("ground_truth")
        if gt is not None:
            return str(gt).strip()

    # Fallbacks for possible schema variants.
    for key in ["ground_truth", "answer", "solution", "target"]:
        if key in example and example[key] is not None:
            return str(example[key]).strip()

    return ""


def boxed_answer(ans):
    ans = str(ans).strip()

    if ans == "":
        return ""

    if ans.startswith("\\boxed"):
        return ans

    return f"\\boxed{{{ans}}}"


def normalize_dapo_example(example):
    problem = clean_dapo_problem(get_dapo_prompt_content(example))
    answer = get_dapo_ground_truth(example)

    meta = get_math_meta_from_problem(problem)

    return {
        "problem": problem,
        "solution": boxed_answer(answer),
        "source": str(example.get("data_source", "dapo")),
        "dapo_level": meta["level"] if meta is not None else "UNMATCHED",
        "dapo_type": meta["type"] if meta is not None else "UNMATCHED",
        "dapo_matched_math": meta is not None,
    }


def normalize_dapo_dataset(dataset):
    return dataset.map(
        normalize_dapo_example,
        remove_columns=dataset.column_names,
        desc="Normalizing DAPO-Math-17k",
    )


def filter_dapo_train(example):
    """
    DAPO train condition:
    - problem must match MATH metadata
    - matched MATH level/type must be inside TRAIN_LEVELS/TRAIN_TYPES
    """
    if not example.get("dapo_matched_math", False):
        return False

    return (
        example["dapo_level"] in TRAIN_LEVELS
        and example["dapo_type"] in TRAIN_TYPES
    )


def filter_dapo_ood(example):
    """
    Everything excluded from DAPO training becomes OOD candidate.
    Non-extractable answers are removed later by filter_extractable_answers().
    """
    return not filter_dapo_train(example)

def filter_extractable_answers(dataset, dataset_name):
    """
    Keep only rows whose final answer can be extracted by the dataset-agnostic
    extractor in dataset_formatting.py.

    Important:
    - We do NOT call format_rlvr_example(..., dataset_name=...).
    - dataset_name is only used for logging.
    """
    before = len(dataset)

    dataset = dataset.filter(
        lambda example: is_extractable_example(example),
        desc=f"Filtering non-extractable answers [{dataset_name}]",
    )

    after = len(dataset)
    dropped = before - after

    print(
        f"[Answer extraction] dataset={dataset_name} "
        f"kept={after}/{before} dropped={dropped}"
    )

    return dataset


def format_dataset(dataset, tokenizer, dataset_name, mode):
    """
    Applies prompt/completion formatting after answer-extraction filtering.
    """
    mode = mode.lower()

    dataset = filter_extractable_answers(dataset, dataset_name)

    if len(dataset) == 0:
        print(f"[Warning] dataset={dataset_name} is empty after filtering.")
        return None

    if "prompt_id" not in dataset.column_names:
        dataset = add_prompt_ids(dataset, seed=SEED)

    remove_columns = dataset.column_names

    if mode == "sft":
        return dataset.map(
            lambda example: format_sft_example(
                example,
                tokenizer=tokenizer,
                prompt_id=example["prompt_id"],
            ),
            remove_columns=remove_columns,
            desc=f"Formatting SFT dataset [{dataset_name}]",
        )

    if mode == "rlvr":
        return dataset.map(
            lambda example: format_rlvr_example(
                example,
                prompt_id=example["prompt_id"],
            ),
            remove_columns=remove_columns,
            desc=f"Formatting RLVR dataset [{dataset_name}]",
        )

    raise ValueError(f"Unknown mode: {mode}")


def build_mixed_dataset(parts, tokenizer, training):
    formatted_sets = []

    for part in parts:
        sampled = sample_dataset(
            part["dataset"],
            part.get("n_samples"),
        )

        formatted = format_dataset(
            sampled,
            tokenizer,
            dataset_name=part["dataset_name"],
            mode=training,
        )

        if formatted is not None and len(formatted) > 0:
            formatted_sets.append(formatted)

    if not formatted_sets:
        raise ValueError("No dataset rows left after filtering/formatting.")

    mixed = concatenate_datasets(formatted_sets)
    mixed = mixed.shuffle(seed=SEED)

    return split_train_val(mixed)

def build_train_val_dataset(tokenizer, training):
    """
    Main train/val builder.

    SFT:
        Numina only.

    RLVR:
        GSM8K + filtered MATH + filtered DAPO.
    """
    training = training.lower()
    load_dataset_lazily()
    datasets = _DATASETS_CACHE

    if training == "sft":
        numina_train = sample_dataset(datasets["numina"]["train"], NUMINA_SAMPLES)
        parts = [
            {
                "dataset": numina_train,
                "dataset_name": "numina",
                "n_samples": None,
            },
        ]

        return build_mixed_dataset(
            parts=parts,
            tokenizer=tokenizer,
            training="sft",
        )

    if training == "rlvr":
        gsm_train = datasets["gsm8k"]["train"]

        math_train = datasets["math"]["train"].filter(
            lambda x: filter_math(x, TRAIN_LEVELS, TRAIN_TYPES),
            desc="Filtering MATH train",
        )

        dapo_norm = normalize_dapo_dataset(datasets["dapo"])
        dapo_train = dapo_norm.filter(
            filter_dapo_train,
            desc="Filtering DAPO train using MATH metadata",
        )

        print(
            f"[DAPO split] train={len(dapo_train)} "
            f"excluded_to_ood_candidate={len(dapo_norm) - len(dapo_train)}"
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
                "dataset": dapo_train,
                "dataset_name": "dapo_train",
                "n_samples": 1_500,
            },
        ]

        return build_mixed_dataset(
            parts=parts,
            tokenizer=tokenizer,
            training="rlvr",
        )

    raise ValueError(f"Unknown training mode: {training}")

def build_ood_eval_dataset(
    tokenizer,
    mode="rlvr",
    save_local=True,
    save_dir=OOD_SAVE_DIR,
    overwrite=True,
    load_if_exists=False,
    check_leakage=True,
):
    """
    Builds the full OOD evaluation dataset.

    OOD sources:
    - GSM8K official test split
    - MATH-lighteval OOD split by level/type
    - DAPO rows excluded from the DAPO train filter

    Important:
    DAPO has no official test split here. DAPO OOD means:
        DAPO rows such that filter_dapo_train(example) == False

    Leakage handling:
    - Build the full RLVR raw train pool.
    - Compute normalized problem fingerprints.
    - Remove from OOD every row whose fingerprint appears in train.
    - Assert that no exact normalized train/OOD overlap remains.
    - Format the cleaned OOD dataset.
    - Optionally save it locally.
    """
    mode = mode.lower()

    if load_if_exists and os.path.exists(save_dir):
        print(f"[OOD load] loading existing dataset from: {save_dir}")
        return load_from_disk(save_dir)

    load_dataset_lazily()
    datasets = _DATASETS_CACHE
    # ============================================================
    # 1. Build raw RLVR train pool
    # ============================================================

    gsm_train = datasets["gsm8k"]["train"]

    math_train = datasets["math"]["train"].filter(
        lambda x: filter_math(x, TRAIN_LEVELS, TRAIN_TYPES),
        desc="Filtering MATH train for leakage check",
    )

    dapo_norm = normalize_dapo_dataset(datasets["dapo"])
    dapo_train = dapo_norm.filter(
        filter_dapo_train,
        desc="Filtering DAPO train for leakage check",
    )

    print(f"[Train raw] gsm8k_train={len(gsm_train)}")
    print(f"[Train raw] math_train={len(math_train)}")
    print(f"[Train raw] dapo_train={len(dapo_train)}")

    train_raw_parts = [
        add_leakage_metadata(gsm_train, "gsm8k_train"),
        add_leakage_metadata(math_train, "math_train"),
        add_leakage_metadata(dapo_train, "dapo_train"),
    ]

    train_fingerprints = collect_train_fingerprints(train_raw_parts)

    print(f"[Leakage index] train_unique_fingerprints={len(train_fingerprints)}")

    # ============================================================
    # 2. Build raw OOD candidates
    # ============================================================

    gsm_tst_raw = datasets["gsm8k"]["test"]
    math_ood_raw = datasets["math"]["train"].filter(
        lambda x: filter_math(x, OOD_LEVELS, OOD_TYPES),
        desc="Filtering MATH OOD",
    )

    dapo_ood_raw = dapo_norm.filter(
        filter_dapo_ood,
        desc="Filtering DAPO OOD: excluded from train",
    )

    print(f"[OOD raw before leakage removal] gsm8k_test={len(gsm_tst_raw)}")
    print(f"[OOD raw before leakage removal] math_ood={len(math_ood_raw)}")
    print(f"[OOD raw before leakage removal] dapo_ood={len(dapo_ood_raw)}")

    # ============================================================
    # 3. Remove train/OOD leakage from every OOD source
    # ============================================================

    gsm_tst = remove_train_overlap_from_ood(
        dataset=gsm_tst_raw,
        train_fingerprints=train_fingerprints,
        source_name="gsm8k_test",
    )

    math_ood = remove_train_overlap_from_ood(
        dataset=math_ood_raw,
        train_fingerprints=train_fingerprints,
        source_name="math_ood",
    )

    dapo_ood = remove_train_overlap_from_ood(
        dataset=dapo_ood_raw,
        train_fingerprints=train_fingerprints,
        source_name="dapo_ood",
    )

    print(f"[OOD raw after leakage removal] gsm8k_test={len(gsm_tst)}")
    print(f"[OOD raw after leakage removal] math_ood={len(math_ood)}")
    print(f"[OOD raw after leakage removal] dapo_ood={len(dapo_ood)}")

    ood_raw_parts = [
        gsm_tst,
        math_ood,
        dapo_ood,
    ]

    # ============================================================
    # 4. Final train/OOD leakage assertion
    # ============================================================

    if check_leakage:
        leakage_report = assert_no_train_ood_overlap(
            train_raw_parts=train_raw_parts,
            ood_raw_parts=ood_raw_parts,
        )
        print(f"[Leakage check passed] {leakage_report}")

    # ============================================================
    # 5. Format OOD dataset
    # ============================================================

    parts = [
        {
            "dataset": gsm_tst,
            "dataset_name": "gsm8k_test",
            "n_samples": None,
        },
        {
            "dataset": math_ood,
            "dataset_name": "math_ood",
            "n_samples": None,
        },
        {
            "dataset": dapo_ood,
            "dataset_name": "dapo_ood",
            "n_samples": 1_200,
        },
    ]

    formatted_sets = []

    for part in parts:
        sampled = sample_dataset(
            part["dataset"],
            part.get("n_samples"),
        )

        formatted = format_dataset(
            sampled,
            tokenizer,
            dataset_name=part["dataset_name"],
            mode=mode,
        )

        if formatted is not None and len(formatted) > 0:
            formatted = add_source_column(
                formatted,
                source_name=part["dataset_name"],
            )

            formatted_sets.append(formatted)

            print(
                f"[OOD formatted] source={part['dataset_name']} "
                f"rows={len(formatted)}"
            )
        else:
            print(
                f"[OOD formatted] source={part['dataset_name']} "
                f"rows=0"
            )

    if not formatted_sets:
        raise ValueError("No OOD eval rows left after filtering/formatting.")

    # ============================================================
    # 6. Concatenate, shuffle, save
    # ============================================================

    ood = concatenate_datasets(formatted_sets)
    ood = ood.shuffle(seed=SEED)

    print(f"[OOD final] rows={len(ood)} columns={ood.column_names}")

    if save_local:
        ood = save_dataset_to_local(
            dataset=ood,
            save_dir=save_dir,
            overwrite=overwrite,
            save_jsonl=True,
        )

    return ood
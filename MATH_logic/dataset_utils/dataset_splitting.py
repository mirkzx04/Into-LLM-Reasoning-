import os
import sys
import re
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from datasets import load_dataset, concatenate_datasets

from MATH_logic.dataset_utils.dataset_formatting import (
    format_sft_example,
    format_rlvr_example,
    add_prompt_ids,
)


SEED = 42
TEST_SIZE = 0.01

NUMINA_SAMPLES = 80_000
DAPO_SAMPLES = None

TRAIN_LEVELS = ["Level 1", "Level 2", "Level 3", "Level 4"]
TRAIN_TYPES = ["Intermediate Algebra", "Number Theory", "Prealgebra", "Precalculus"]

OOD_LEVELS = ["Level 5"]
OOD_TYPES = ["Algebra", "Counting & Probability", "Geometry"]


numina_set = load_dataset("AI-MO/NuminaMath-CoT")

gsm8k_set = (
    load_dataset("openai/gsm8k", "main")
    .rename_columns({"question": "problem", "answer": "solution"})
)

math_set = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default")

dapo_set = load_dataset(
    "BytedTsinghua-SIA/DAPO-Math-17k",
    split="train",
)


def normalize_txt(x):
    return re.sub(r"\s+", " ", str(x)).strip()


def normalize_for_lookup(x):
    x = str(x)
    x = x.replace("\r\n", "\n")
    x = re.sub(r"\s+", " ", x).strip()
    x = x.replace("\\left", "").replace("\\right", "")
    return x


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


def split_train_val(dataset):
    split = dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
    return split["train"], split["test"]


def sample_dataset(dataset, n_samples=None):
    dataset = dataset.shuffle(seed=SEED)

    if n_samples is None:
        return dataset

    n_samples = min(int(n_samples), len(dataset))
    return dataset.select(range(n_samples))


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

    if ans.startswith("\\boxed"):
        return ans

    return f"\\boxed{{{ans}}}"


def get_math_meta_from_problem(problem):
    return MATH_LOOKUP.get(normalize_for_lookup(problem))


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
    Non-extractable answers are still removed later by format_dataset().
    """
    return not filter_dapo_train(example)


def is_extractable_after_formatting(example, dataset_name):
    """
    Uses the same formatter used by training. If formatting produces UNKNOWN
    or raises, the sample is removed before train/eval.
    """
    try:
        formatted = format_rlvr_example(
            example,
            dataset_name=dataset_name,
            prompt_id=0,
        )
        sol = str(formatted["solution"])
        return "UNKNOWN" not in sol and "\\boxed{}" not in sol
    except Exception:
        return False


def filter_extractable_answers(dataset, dataset_name):
    before = len(dataset)

    dataset = dataset.filter(
        lambda example: is_extractable_after_formatting(example, dataset_name),
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

    dataset = filter_extractable_answers(dataset, dataset_name)

    if len(dataset) == 0:
        return None

    if "prompt_id" not in dataset.column_names:
        dataset = add_prompt_ids(dataset, seed=SEED)

    remove_columns = dataset.column_names

    if mode == "sft":
        return dataset.map(
            lambda example: format_sft_example(
                example,
                tokenizer=tokenizer,
                dataset_name=dataset_name,
                prompt_id=example["prompt_id"],
            ),
            remove_columns=remove_columns,
            desc=f"Formatting SFT dataset [{dataset_name}]",
        )

    if mode == "rlvr":
        return dataset.map(
            lambda example: format_rlvr_example(
                example,
                dataset_name=dataset_name,
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

    if training == "sft":
        numina_train = sample_dataset(numina_set["train"], NUMINA_SAMPLES)

        parts = [
            {
                "dataset": numina_train,
                "dataset_name": "math",
                "n_samples": None,
            },
        ]

        return build_mixed_dataset(
            parts=parts,
            tokenizer=tokenizer,
            training="sft",
        )

    if training == "rlvr":
        gsm_train = gsm8k_set["train"]

        math_train = math_set["train"].filter(
            lambda x: filter_math(x, TRAIN_LEVELS, TRAIN_TYPES),
            desc="Filtering MATH train",
        )

        dapo_norm = normalize_dapo_dataset(dapo_set)

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
                "dataset_name": "math",
                "n_samples": DAPO_SAMPLES,
            },
        ]

        return build_mixed_dataset(
            parts=parts,
            tokenizer=tokenizer,
            training="rlvr",
        )

    raise ValueError(f"Unknown training mode: {training}")


def build_ood_eval_dataset(tokenizer, mode="rlvr"):
    """
    OOD eval builder.

    Includes:
    - GSM8K test
    - MATH OOD split
    - DAPO excluded from RLVR training
    """

    gsm_tst = gsm8k_set["test"]

    math_ood = math_set["train"].filter(
        lambda x: filter_math(x, OOD_LEVELS, OOD_TYPES),
        desc="Filtering MATH OOD",
    )

    dapo_norm = normalize_dapo_dataset(dapo_set)

    dapo_ood = dapo_norm.filter(
        filter_dapo_ood,
        desc="Filtering DAPO OOD: excluded from train",
    )

    print(f"[DAPO OOD] size={len(dapo_ood)}")

    parts = [
        {
            "dataset": gsm_tst,
            "dataset_name": "gsm8k",
            "n_samples": None,
        },
        {
            "dataset": math_ood,
            "dataset_name": "math",
            "n_samples": None,
        },
        {
            "dataset": dapo_ood,
            "dataset_name": "math",
            "n_samples": None,
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

        if ds is not None and len(ds) > 0:
            formatted_sets.append(ds)

    if not formatted_sets:
        raise ValueError("No OOD eval rows left after filtering/formatting.")

    return concatenate_datasets(formatted_sets).shuffle(seed=SEED)
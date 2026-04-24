
import os 
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from datasets import load_dataset, concatenate_datasets

from MATH_logic.dataset_utils.dataset_formatting import format_sft_example, format_rlvr_example

NUMINA_PERCENT = 0.1

SEED = 42
TEST_SIZE = 0.1

NUMINA_SAMPLES = 80_000

T1_TOTAL_SAMPLES = 20_000
T2_TOTAL_SAMPLES = 20_000
T3_TOTAL_SAMPLES = 20_000

numina_set_train = load_dataset("AI-MO/NuminaMath-CoT", split = "train")
gsm8k_set_train = load_dataset("openai/gsm8k", "main", split = "train")
math_set_train = load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split = "train")

def get_math_lvl(levels) : 
    return  math_set_train.filter(lambda x : x["level"] in levels)

def split_train_val(dataset) : 
    split = dataset.train_test_split(test_size=TEST_SIZE, seed=SEED)
    return split["train"], split["test"]

def sample_dataset(dataset, n_samples):
    n_samples = min(n_samples, len(dataset))

    return (
        dataset
        .shuffle(seed=SEED)
        .select(range(n_samples))
    )

def format_dataset(dataset, tokenizer, dataset_name, type_training):
    if type_training == "sft" :
        return dataset.map(
            lambda example: format_sft_example(
                example,
                tokenizer,
                dataset_name=dataset_name
            ),
            remove_columns=dataset.column_names
        )
    if type_training == "rlvr" :  
        return dataset.map(
            lambda example: format_rlvr_example(
                example,
                tokenizer,
                dataset_name=dataset_name
            ),
            remove_columns=dataset.column_names
        )

def build_mixed_dataset(parts, tokenizer, type_training) :
    """
        parts format:

        [
            {
                "dataset": gsm8k_set_train,
                "dataset_name": "gsm8k",
                "n_samples": 14000,
            },
            {
                "dataset": math_lvl_1_2,
                "dataset_name": "math",
                "n_samples": 6000,
            },
        ]
    """
     
    formatted_sets = []
    for part in parts : 
        sampled = sample_dataset(part["dataset"], part["n_samples"])

        formatted = format_dataset(
            sampled,
            tokenizer,
            dataset_name=part["dataset_name"],
            type_trn=type_training
        )

        formatted_sets.append(formatted)

    mixed = concatenate_datasets(formatted_sets)
    mixed = mixed.shuffle(seed=SEED)

    return split_train_val(mixed)
    

def build_numina_train(tokenizer):
    numina_set = sample_dataset(numina_set_train, NUMINA_SAMPLES)
    numina_set = format_dataset(numina_set, tokenizer, dataset_name="numina")

    return split_train_val(numina_set)

def build_t1_set(tokenizer, type_training):
    total = T1_TOTAL_SAMPLES

    math_lvl_1_2 = get_math_lvl(["Level 1", "Level 2"])

    return build_mixed_dataset(
        parts=[
            {
                "dataset": gsm8k_set_train,
                "dataset_name": "gsm8k",
                "n_samples": int(total * 0.70),
            },
            {
                "dataset": math_lvl_1_2,
                "dataset_name": "math",
                "n_samples": int(total * 0.30),
            },
        ],
        tokenizer=tokenizer, 
        type_training = type_training
    )

def build_t2_set(tokenizer, type_training):
    total = T2_TOTAL_SAMPLES

    math_lvl_1_2 = get_math_lvl(["Level 1", "Level 2"])
    math_lvl_3 = get_math_lvl(["Level 3"])

    return build_mixed_dataset(
        parts=[
            {
                "dataset": gsm8k_set_train,
                "dataset_name": "gsm8k",
                "n_samples": int(total * 0.20),
            },
            {
                "dataset": math_lvl_1_2,
                "dataset_name": "math",
                "n_samples": int(total * 0.60),
            },
            {
                "dataset": math_lvl_3,
                "dataset_name": "math",
                "n_samples": int(total * 0.20),
            },
        ],
        tokenizer=tokenizer,
        type_training = type_training
    )

def build_t3_set(tokenizer, type_training):
    total = T3_TOTAL_SAMPLES

    math_lvl_1_2 = get_math_lvl(["Level 1", "Level 2"])
    math_lvl_3 = get_math_lvl(["Level 3"])
    math_lvl_4 = get_math_lvl(["Level 4"])

    return build_mixed_dataset(
        parts=[
            {
                "dataset": gsm8k_set_train,
                "dataset_name": "gsm8k",
                "n_samples": int(total * 0.10),
            },
            {
                "dataset": math_lvl_1_2,
                "dataset_name": "math",
                "n_samples": int(total * 0.45),
            },
            {
                "dataset": math_lvl_3,
                "dataset_name": "math",
                "n_samples": int(total * 0.35),
            },
            {
                "dataset": math_lvl_4,
                "dataset_name": "math",
                "n_samples": int(total * 0.10),
            },
        ],
        tokenizer=tokenizer, 
        type_training=type_training
    )

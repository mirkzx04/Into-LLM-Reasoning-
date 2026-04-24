import re

# Regex to match LaTeX \boxed{} commands
BOXED_RE = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", re.DOTALL)


def split_reasoning_steps(reasoning_text: str):
    """Splits a continuous block of reasoning text into discrete logical steps."""
    reasoning_text = str(reasoning_text).strip()
    reasoning_text = reasoning_text.replace("\r\n", "\n")
    reasoning_text = re.sub(r"\n{3,}", "\n\n", reasoning_text)

    # Split into initial chunks based on empty lines
    chunks = [c.strip() for c in re.split(r"\n\s*\n", reasoning_text) if c.strip()]

    steps = []
    for chunk in chunks:
        # Keep short chunks as single steps
        if len(chunk) < 120:
            steps.append(chunk)
            continue

        # Further split long chunks based on punctuation followed by uppercase letters
        substeps = re.split(r"(?<=[.!?])\s+(?=[A-Z\\(])", chunk)
        substeps = [s.strip() for s in substeps if s.strip()]
        steps.extend(substeps if substeps else [chunk])

    return steps if steps else ["We solve the problem step by step."]


def extract_last_boxed(text: str):
    """Finds and returns the content of the last \boxed{} occurrence in the text."""
    matches = BOXED_RE.findall(str(text))
    return matches[-1].strip() if matches else None


def extract_gsm8k_answer(answer: str):
    """
    Parses GSM8K solutions to separate the reasoning process from the final answer.
    GSM8K format:
    reasoning...
    #### final_answer
    """
    answer = str(answer).strip()

    # Handle the standard GSM8K format marked by '####'
    if "####" in answer:
        reasoning, final = answer.rsplit("####", 1)
        return reasoning.strip(), final.strip()

    # Fallback: look for a boxed answer if the standard delimiter is missing
    boxed = extract_last_boxed(answer)
    if boxed is not None:
        reasoning = BOXED_RE.sub(boxed, answer).strip()
        return reasoning, boxed

    return answer, None


def extract_math_answer(solution: str):
    """
    Parses MATH or NuminaMath-CoT dataset solutions.
    Prefer the last \\boxed{} as final answer and strips out alternate variations.
    """
    solution = str(solution).strip()

    # Exclude alternative solutions commonly indicated by "- OR -"
    solution = re.split(r"\n\s*-\s*OR\s*-\s*\n", solution, maxsplit=1)[0].strip()

    final_answer = extract_last_boxed(solution)

    # Separate the text reasoning from the extracted boxed final answer
    if final_answer is not None:
        reasoning = BOXED_RE.sub("", solution).strip()
        return reasoning, final_answer

    return solution, None


def build_tagged_completion(reasoning: str, final_answer: str):
    """Wraps reasoning and the final answer within specific XML-like tags."""
    steps = split_reasoning_steps(reasoning)

    # Wrap each split step into a custom <reasoning:step> tag
    tagged_steps = "\n".join(
        f"<reasoning:step>{step}</reasoning:step>"
        for step in steps
    )

    if final_answer is None or str(final_answer).strip() == "":
        final_answer = "UNKNOWN"

    return (
        f"{tagged_steps}\n"
        f"<answer>\\boxed{{{str(final_answer).strip()}}}</answer>"
    )


def convert_solution_to_tagged_completion(solution: str, dataset_name: str = "math"):
    """Routes the parsing operation based on the dataset type to generate a tagged completion string."""
    dataset_name = dataset_name.lower()

    if dataset_name in ["gsm8k", "openai/gsm8k"]:
        reasoning, final_answer = extract_gsm8k_answer(solution)
    else:
        reasoning, final_answer = extract_math_answer(solution)

    return build_tagged_completion(reasoning, final_answer)


def build_math_prompt(problem: str):
    """Formats the input prompt to instruct the model on the expected output structure."""
    return (
        "Solve this math problem step by step.\n"
        "Write each reasoning step inside <reasoning:step>...</reasoning:step> tags.\n"
        "At the end, put the final answer inside <answer>\\boxed{...}</answer>.\n"
        f"Problem: {problem}\n"
    )


def format_sft_example(example, tokenizer=None, dataset_name: str = "math"):
    """
    Transforms a generic dataset example into the format required for SFTTrainer:
    {
        "prompt": ...,
        "completion": ...
    }
    """

    dataset_name = dataset_name.lower()

    # Map column names depending on the source dataset
    if dataset_name in ["gsm8k", "openai/gsm8k"]:
        problem = example["question"]
        raw_solution = example["answer"]

    elif dataset_name in ["math", "math-lighteval", "digitallearninggmbh/math-lighteval"]:
        problem = example["problem"]
        raw_solution = example["solution"]

    elif dataset_name in ["numina", "numinamath", "ai-mo/numinamath-cot"]:
        problem = example["problem"]
        raw_solution = example["solution"]

    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    # Generate model target
    completion = convert_solution_to_tagged_completion(
        raw_solution,
        dataset_name=dataset_name
    )

    # Append end-of-sequence token if available
    if tokenizer is not None and tokenizer.eos_token is not None:
        completion += tokenizer.eos_token

    return {
        "prompt": build_math_prompt(problem),
        "completion": completion,
    }


def format_rlvr_example(example, dataset_name: str = "math"):
    """
    Transforms a generic dataset example into the format required for GRPOTrainer:
    {
        "prompt": ...,
        "solution": ...
    }

    The solution is normalized to containing <answer>\\boxed{...}</answer>
    so that rewards functions can easily extract the Ground Truth answer.
    """

    dataset_name = dataset_name.lower()

    # Dispatch to specific logic according to dataset mappings
    if dataset_name in ["gsm8k", "openai/gsm8k"]:
        problem = example["question"]
        _, final_answer = extract_gsm8k_answer(example["answer"])

    elif dataset_name in ["math", "math-lighteval", "digitallearninggmbh/math-lighteval"]:
        problem = example["problem"]
        _, final_answer = extract_math_answer(example["solution"])

    elif dataset_name in ["numina", "numinamath", "ai-mo/numinamath-cot"]:
        problem = example["problem"]
        _, final_answer = extract_math_answer(example["solution"])

    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    if final_answer is None:
        final_answer = "UNKNOWN"

    return {
        "prompt": build_math_prompt(problem),
        "solution": f"<answer>\\boxed{{{final_answer}}}</answer>",
    }


def filter_math_level(example, allowed_levels):
    """
    Filters rows based on difficulty level values.
    For MATH-lighteval, level formats can be like 'Level 2' or integer strings.
    """
    if "level" not in example:
        return False

    level_text = str(example["level"])
    match = re.search(r"\d+", level_text)

    if match is None:
        return False

    return int(match.group()) in allowed_levels
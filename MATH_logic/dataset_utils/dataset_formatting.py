import re
import random
from dataclasses import dataclass
from typing import Any, Optional, Tuple

try:
    from math_verify import parse
except Exception:
    parse = None


PROMPT_TEMPLATES = [
    (
        "Solve this math problem step by step.\n"
        "Write each reasoning step inside <reasoning:step>...</reasoning:step> tags.\n"
        "At the end, put the final answer inside <answer>\\boxed{{...}}</answer>.\n"
        "Problem: {problem}\n"
    ),
    (
        "You are given a math problem. Reason carefully and solve it step by step.\n"
        "Every reasoning step must be enclosed in <reasoning:step>...</reasoning:step> tags.\n"
        "Return the final answer only inside <answer>\\boxed{{...}}</answer>.\n"
        "Problem: {problem}\n"
    ),
    (
        "Work through the following problem using explicit mathematical reasoning.\n"
        "Use one or more <reasoning:step>...</reasoning:step> blocks for the solution.\n"
        "Finish with <answer>\\boxed{{...}}</answer> and nothing after it.\n"
        "Problem: {problem}\n"
    ),
    (
        "Find the answer to the math problem below.\n"
        "Show the reasoning using <reasoning:step>...</reasoning:step> tags.\n"
        "The final result must be written as <answer>\\boxed{{...}}</answer>.\n"
        "Problem: {problem}\n"
    ),
    (
        "Solve the following mathematical question.\n"
        "Put each logical step inside <reasoning:step>...</reasoning:step>.\n"
        "Put the final answer inside <answer>\\boxed{{...}}</answer>.\n"
        "Problem: {problem}\n"
    ),
]


ANSWER_TAG_RE = re.compile(r"<answer>\s*([\s\S]*?)\s*</answer>", re.DOTALL | re.IGNORECASE)

FINAL_MARKER_RE = re.compile(
    r"(?i)"
    r"(?:"
    r"final\s+answer|"
    r"the\s+answer|"
    r"answer|"
    r"ans|"
    r"result|"
    r"therefore|"
    r"hence"
    r")"
    r"\s*(?:is|=|:)?\s*"
    r"(?P<ans>[^\n]+)"
)

MATH_ENV_RE = re.compile(
    r"\$\$(?P<ddollar>[\s\S]+?)\$\$"
    r"|\$(?P<dollar>[^$\n]+?)\$"
    r"|\\\((?P<paren>[\s\S]+?)\\\)"
    r"|\\\[(?P<bracket>[\s\S]+?)\\\]",
    re.DOTALL
)

NUMBER_RE = re.compile(
    r"(?<![A-Za-z])"
    r"[-+]?"
    r"(?:"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+)?"
    r"|"
    r"\.\d+"
    r")"
    r"%?"
)

PROBLEM_COLUMNS = [
    "problem",
    "question",
    "query",
    "instruction",
    "input",
    "prompt",
]

SOLUTION_COLUMNS = [
    "solution",
    "answer",
    "response",
    "output",
    "completion",
    "target",
    "label",
]

DIRECT_ANSWER_COLUMNS = [
    "final_answer",
    "short_answer",
    "answer",
    "target",
    "ground_truth",
    "ground_truth_answer",
    "gt_answer",
    "correct_answer",
    "label",
]


@dataclass
class AnswerExtraction:
    answer: str
    method: str
    span: Optional[Tuple[int, int]] = None


def add_prompt_ids(dataset, seed=42):
    n = len(dataset)
    n_prompts = len(PROMPT_TEMPLATES)

    prompt_ids = [i % n_prompts for i in range(n)]

    rng = random.Random(seed)
    rng.shuffle(prompt_ids)

    return dataset.add_column("prompt_id", prompt_ids)


def get_first_existing_field(example: dict, columns):
    for col in columns:
        if col in example and example[col] is not None:
            value = str(example[col]).strip()
            if value:
                return value
    return None


def get_problem_text(example: dict):
    problem = get_first_existing_field(example, PROBLEM_COLUMNS)
    if problem is None:
        raise KeyError(f"No problem column found. Available columns: {list(example.keys())}")
    return problem


def get_solution_text(example: dict):
    solution = get_first_existing_field(example, SOLUTION_COLUMNS)
    if solution is None:
        raise KeyError(f"No solution/answer column found. Available columns: {list(example.keys())}")
    return solution


def normalize_solution_text(text: str):
    text = str(text).strip()
    text = text.replace("\r\n", "\n")

    # Numina/MATH-style alternative solutions: keep first canonical branch.
    text = re.split(r"\n\s*-\s*OR\s*-\s*\n", text, maxsplit=1)[0].strip()

    return text


def strip_math_wrappers(ans: str):
    ans = ans.strip()

    wrappers = [
        (r"^\$(.*)\$$", 1),
        (r"^\\\((.*)\\\)$", 1),
        (r"^\\\[(.*)\\\]$", 1),
    ]

    for pattern, group_id in wrappers:
        m = re.match(pattern, ans, flags=re.DOTALL)
        if m:
            ans = m.group(group_id).strip()

    ans = ans.replace("\\left", "").replace("\\right", "")
    return ans.strip()


def clean_candidate(ans: Any):
    if ans is None:
        return None

    ans = str(ans).strip()
    if not ans:
        return None

    ans = ans.replace("\r\n", "\n").strip()
    ans = re.sub(r"^#+\s*", "", ans).strip()
    ans = re.sub(r"^####\s*", "", ans).strip()
    ans = strip_math_wrappers(ans)

    # Remove common textual prefixes.
    ans = re.sub(
        r"(?i)^(?:the\s+)?(?:final\s+)?(?:answer|result|value)\s*(?:is|=|:)\s*",
        "",
        ans,
    ).strip()

    # If the candidate still contains a full sentence, take the first line.
    ans = ans.split("\n", 1)[0].strip()

    # Remove trailing final punctuation but preserve math delimiters.
    ans = ans.strip(" \t")
    ans = re.sub(r"[\.;:,]\s*$", "", ans).strip()

    # Remove leading/trailing markdown emphasis.
    ans = ans.strip("*_` ")

    return ans or None


def safe_parse(ans: str):
    if parse is None:
        return [ans] if ans else []

    try:
        return parse(
            ans,
            fallback_mode="no_fallback",
            extraction_mode="any_match",
            parsing_timeout=0,
        )
    except TypeError:
        return parse(ans)
    except Exception:
        return []


def is_parseable_answer(ans: Any):
    ans = clean_candidate(ans)
    if ans is None:
        return False

    parsed = safe_parse(ans)
    return bool(parsed)


def _extract_braced_arg(text: str, open_brace_idx: int):
    depth = 0
    start = None

    for i in range(open_brace_idx, len(text)):
        ch = text[i]

        if ch == "{":
            depth += 1
            if depth == 1:
                start = i + 1

        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return text[start:i], (open_brace_idx, i + 1)

            if depth < 0:
                return None, None

    return None, None


def find_latex_command_args(text: str, commands=("boxed", "fbox")):
    """
    Robust extraction for \\boxed{...}, including nested braces.
    Regex-only boxed extraction often breaks on nested LaTeX.
    """
    out = []

    for cmd in commands:
        pattern = re.compile(rf"\\{cmd}\s*\{{")
        for m in pattern.finditer(text):
            open_brace_idx = text.find("{", m.start())
            content, brace_span = _extract_braced_arg(text, open_brace_idx)
            if content is not None and brace_span is not None:
                out.append((content.strip(), (m.start(), brace_span[1])))

    out.sort(key=lambda x: x[1][1])
    return out


def extract_last_boxed(text: str):
    boxes = find_latex_command_args(str(text), commands=("boxed", "fbox"))
    if not boxes:
        return None

    ans = clean_candidate(boxes[-1][0])
    return ans


def _candidate_from_answer_tag(text: str):
    matches = list(ANSWER_TAG_RE.finditer(text))

    for m in reversed(matches):
        content = m.group(1).strip()

        boxed = find_latex_command_args(content)
        if boxed:
            candidate = clean_candidate(boxed[-1][0])
        else:
            candidate = clean_candidate(content)

        if is_parseable_answer(candidate):
            return AnswerExtraction(candidate, "answer_tag", (m.start(), m.end()))

    return None


def _candidate_from_boxed(text: str):
    boxes = find_latex_command_args(text)

    for candidate, span in reversed(boxes):
        candidate = clean_candidate(candidate)
        if is_parseable_answer(candidate):
            return AnswerExtraction(candidate, "boxed", span)

    return None


def _candidate_from_gsm_hash(text: str):
    if "####" not in text:
        return None

    tail = text.rsplit("####", 1)[1]
    candidates = [tail.strip(), tail.strip().split("\n", 1)[0].strip()]

    for candidate in candidates:
        candidate = clean_candidate(candidate)
        if is_parseable_answer(candidate):
            start = text.rfind("####")
            return AnswerExtraction(candidate, "gsm8k_hash", (start, len(text)))

    return None


def _candidate_from_final_markers(text: str):
    matches = list(FINAL_MARKER_RE.finditer(text))

    for m in reversed(matches):
        raw = m.group("ans").strip()

        # Try complete marker tail first.
        candidates = [raw]

        # Then try math environments inside the marker tail.
        candidates.extend(extract_math_env_candidates(raw))

        # Then try last numeric candidate inside the marker tail.
        nums = NUMBER_RE.findall(raw)
        if nums:
            candidates.append(nums[-1])

        for candidate in candidates:
            candidate = clean_candidate(candidate)
            if is_parseable_answer(candidate):
                return AnswerExtraction(candidate, "final_marker", (m.start(), m.end()))

    return None


def extract_math_env_candidates(text: str):
    candidates = []

    for m in MATH_ENV_RE.finditer(text):
        for name in ["ddollar", "dollar", "paren", "bracket"]:
            value = m.groupdict().get(name)
            if value:
                candidates.append(value.strip())

    return candidates


def _candidate_from_math_env(text: str):
    candidates = extract_math_env_candidates(text)

    for candidate in reversed(candidates):
        candidate = clean_candidate(candidate)
        if is_parseable_answer(candidate):
            return AnswerExtraction(candidate, "math_env", None)

    return None


def _candidate_from_last_number(text: str):
    nums = NUMBER_RE.findall(text)
    if not nums:
        return None

    for candidate in reversed(nums):
        candidate = clean_candidate(candidate)
        if is_parseable_answer(candidate):
            return AnswerExtraction(candidate, "last_number", None)

    return None


def _candidate_from_direct_columns(example: Optional[dict]):
    if not example:
        return None

    for col in DIRECT_ANSWER_COLUMNS:
        if col not in example or example[col] is None:
            continue

        raw = str(example[col]).strip()
        if not raw:
            continue

        # If this column is actually a full CoT solution, the direct candidate
        # will fail and normal text extraction will handle it later.
        candidate = clean_candidate(raw)
        if is_parseable_answer(candidate):
            return AnswerExtraction(candidate, f"direct_column:{col}", None)

    return None


def extract_final_answer_auto(solution: str, example: Optional[dict] = None):
    """
    Dataset-agnostic final answer extraction.

    It tries multiple answer conventions and validates candidates with math_verify.
    """
    direct = _candidate_from_direct_columns(example)
    if direct is not None:
        return direct

    text = normalize_solution_text(solution)

    extractors = [
        _candidate_from_answer_tag,
        _candidate_from_boxed,
        _candidate_from_gsm_hash,
        _candidate_from_final_markers,
        _candidate_from_math_env,
        _candidate_from_last_number,
    ]

    for extractor in extractors:
        result = extractor(text)
        if result is not None:
            return result

    return None


def extract_reasoning_and_answer_auto(solution: str, example: Optional[dict] = None):
    text = normalize_solution_text(solution)
    extraction = extract_final_answer_auto(text, example=example)

    if extraction is None:
        return text, None

    reasoning = text

    if extraction.method == "gsm8k_hash" and "####" in text:
        reasoning = text.rsplit("####", 1)[0].strip()

    elif extraction.span is not None:
        start, end = extraction.span
        reasoning = (text[:start] + text[end:]).strip()

    # Remove leftover answer tags if present.
    reasoning = ANSWER_TAG_RE.sub("", reasoning).strip()

    # Avoid leaving a dangling final-answer sentence.
    reasoning = re.sub(
        r"(?i)(?:final\s+answer|the\s+answer|answer|result)\s*(?:is|=|:)\s*$",
        "",
        reasoning,
    ).strip()

    return reasoning, extraction.answer


def split_reasoning_steps(reasoning_text: str):
    reasoning_text = str(reasoning_text).strip()
    reasoning_text = reasoning_text.replace("\r\n", "\n")
    reasoning_text = re.sub(r"\n{3,}", "\n\n", reasoning_text)

    chunks = [c.strip() for c in re.split(r"\n\s*\n", reasoning_text) if c.strip()]

    steps = []
    for chunk in chunks:
        if len(chunk) < 120:
            steps.append(chunk)
            continue

        substeps = re.split(r"(?<=[.!?])\s+(?=[A-Z\\(])", chunk)
        substeps = [s.strip() for s in substeps if s.strip()]
        steps.extend(substeps if substeps else [chunk])

    return steps if steps else ["We solve the problem step by step."]


def build_tagged_completion(reasoning: str, final_answer: str):
    if final_answer is None or str(final_answer).strip() == "":
        return None

    steps = split_reasoning_steps(reasoning)

    tagged_steps = "\n".join(
        f"<reasoning:step>{step}</reasoning:step>"
        for step in steps
    )

    return (
        f"{tagged_steps}\n"
        f"<answer>\\boxed{{{str(final_answer).strip()}}}</answer>"
    )


def convert_solution_to_tagged_completion(solution: str, example: Optional[dict] = None):
    reasoning, final_answer = extract_reasoning_and_answer_auto(solution, example=example)
    return build_tagged_completion(reasoning, final_answer)


def build_math_prompt(problem: str, prompt_id=0):
    template = PROMPT_TEMPLATES[prompt_id % len(PROMPT_TEMPLATES)]
    return template.format(problem=problem)


def is_extractable_example(example: dict):
    try:
        solution = get_solution_text(example)
        extraction = extract_final_answer_auto(solution, example=example)
        return extraction is not None
    except Exception:
        return False


def format_sft_example(example, prompt_id=0, tokenizer=None):
    problem = get_problem_text(example)
    raw_solution = get_solution_text(example)

    completion = convert_solution_to_tagged_completion(
        raw_solution,
        example=example,
    )

    if completion is None:
        raise ValueError("Could not extract final answer from example. Filter it before formatting.")

    if tokenizer is not None and tokenizer.eos_token is not None:
        completion += tokenizer.eos_token

    return {
        "prompt": build_math_prompt(problem, prompt_id),
        "completion": completion,
    }


def format_rlvr_example(example, prompt_id=0):
    problem = get_problem_text(example)
    raw_solution = get_solution_text(example)

    extraction = extract_final_answer_auto(raw_solution, example=example)

    if extraction is None:
        raise ValueError("Could not extract final answer from example. Filter it before formatting.")

    return {
        "prompt": build_math_prompt(problem, prompt_id),
        "solution": f"<answer>\\boxed{{{extraction.answer}}}</answer>",
    }


def filter_math_level(example, allowed_levels):
    if "level" not in example:
        return False

    level_text = str(example["level"])
    match = re.search(r"\d+", level_text)

    if match is None:
        return False

    return int(match.group()) in allowed_levels

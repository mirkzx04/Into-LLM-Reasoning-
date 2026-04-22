import re
from math_verify import parse, verify

# Regular expressions for matching LaTeX \boxed{} patterns and XML-like tags
BOXED_PATTERN = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
BOXED_RE = re.compile(BOXED_PATTERN, re.DOTALL)

ANSWER_TAG_RE = re.compile(r"<answer>\s*([\s\S]*?)\s*</answer>", re.DOTALL)
STEP_BLOCK_RE = re.compile(r"<reasoning:step>[\s\S]*?</reasoning:step>", re.DOTALL)
ANSWER_ONLY_BOXED_RE = re.compile(rf"^\s*{BOXED_PATTERN}\s*$", re.DOTALL)
SPACE_RE = re.compile(r"\s+")

def compact(txt):
    """Removes all whitespace characters from the text."""
    return SPACE_RE.sub("", txt)

def last_boxes(txt):
    """Extracts the content of the last LaTeX \boxed{} expression in the text."""
    matches = BOXED_RE.findall(txt)
    return matches[-1].strip() if matches else None

def safe_verify(pred, gt):
    """
    Safely parses and verifies the predicted answer against the ground truth.
    Returns 1.0 for a match, 0.0 for a mismatch, and -0.5 if parsing fails.
    """
    try:
        pred_parsed = parse(pred)
        gt_parsed = parse(gt)
        return 1.0 if (pred_parsed is not None and gt_parsed is not None and verify(pred_parsed, gt_parsed)) else 0.0
    except Exception:
        return -0.5

def format_reward(completions, **kwargs):
    """
    Assigns a formatting reward based on whether the completion strictly follows
    the expected structural format (e.g., has exactly one <answer> block, valid <reasoning:step> blocks).
    """
    answer_blocks = [ANSWER_TAG_RE.findall(c_txt) for c_txt in completions]
    
    # Check if there is exactly one answer block
    one_answer = [len(blocks) == 1 for blocks in answer_blocks]
    answer_txt = [blocks[0].strip() if len(blocks) == 1 else "" for blocks in answer_blocks]

    # Check if the answer block contains only the boxed expression
    answer_matches = [bool(ANSWER_ONLY_BOXED_RE.fullmatch(a_txt)) for a_txt in answer_txt]

    answer_searches = [ANSWER_TAG_RE.search(c_txt) for c_txt in completions]
    
    # Split the completion into prefix (before <answer>) and suffix (after <answer>)
    prefixes = [c_txt[:m.start()].strip() if m else "" for c_txt, m in zip(completions, answer_searches)]
    suffixes = [c_txt[m.end():].strip() if m else "" for c_txt, m in zip(completions, answer_searches)]

    # Extract all reasoning steps from the prefix
    step_blocks = [STEP_BLOCK_RE.findall(prefix) for prefix in prefixes]
    step_counts = [len(blocks) for blocks in step_blocks]

    # Check if the entire prefix consists only of reasoning steps without rogue text
    prefix_compact = [compact(prefix) for prefix in prefixes]
    blocks_compact = [compact("".join(blocks)) for blocks in step_blocks]

    # Return 1.0 reward only if all structural constraints are perfectly met
    return [
        1.0 if (has_one and ans_ok and suffix == "" and n_steps > 0 and p_cmp == b_cmp) else 0.0
        for has_one, ans_ok, suffix, n_steps, p_cmp, b_cmp
        in zip(one_answer, answer_matches, suffixes, step_counts, prefix_compact, blocks_compact)
    ]

def concise_accuracy_reward(completions, complestion_ids, solution, trainer_state, **kwargs):
    """
    Assigns a reward based on accuracy, with a bonus for more concise completions.
    Bonus strength gradually increases over the training steps.
    """
    def concise_bonus(L, target_len=320, fade_span=256, hard_cap=1024):
        # Applies a penalty if too long, else a bonus that tapers off as length increases
        return (
            -0.25 if L >= hard_cap else
            max(0.0, 1.0 - max(0, L - target_len) / fade_span)
        )

    # Gradually scale formatting bonus/penalty (alpha) from 0.0 to 1.0 over the first 200 steps
    step = 0 if trainer_state is None else trainer_state.global_step
    alpha = min(1.0, step / 200.0)

    answer_blocks = [ANSWER_TAG_RE.findall(c_txt) for c_txt in completions]

    pred_txt = [last_boxes(blocks[0]) if len(blocks) == 1 else None for blocks in answer_blocks]
    gt_txt = [last_boxes(s_txt) for s_txt in solution]

    # Verify accuracy
    correct_flags = [safe_verify(pred, gt) for pred, gt in zip(pred_txt, gt_txt)]
    
    # Calculate length bonus for each completion
    bonuses = [concise_bonus(len(ids), target_len = 512, fade_span = 300, hard_cap = 1024) for ids in complestion_ids]

    # Combine accuracy logic with the scaled bonus
    return[alpha * bonus if correct else 0.0 for correct, bonus in zip(correct_flags, bonuses)]

def acc_reward(completions, solution, **kwargs):
    """Simple exact match accuracy reward by comparing the extracted boxed answers."""
    answer_blocks = [ANSWER_TAG_RE.findall(c_txt) for c_txt in completions]
    pred_txt = [last_boxes(blocks[0]) if len(blocks) == 1 else None for blocks in answer_blocks]
    gt_txt = [last_boxes(s_txt) for s_txt in solution]

    return [safe_verify(pred, gt) for pred, gt in zip(pred_txt, gt_txt)]
import re
from math_verify import parse, verify

BOXED_PATTERN = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
BOXED_RE = re.compile(BOXED_PATTERN, re.DOTALL)

ANSWER_TAG_RE = re.compile(r"<answer>\s*([\s\S]*?)\s*</answer>", re.DOTALL)
STEP_BLOCK_RE = re.compile(r"<reasoning:step>[\s\S]*?</reasoning:step>", re.DOTALL)
ANSWER_ONLY_BOXED_RE = re.compile(rf"^\s*{BOXED_PATTERN}\s*$", re.DOTALL)
SPACE_RE = re.compile(r"\s+")

def compact(txt):
    return SPACE_RE.sub("", txt)

def last_boxes(txt):
    matches = BOXED_RE.findall(txt)
    return matches[-1].strip() if matches else None

def safe_verify(pred, gt):
    try:
        pred_parsed = parse(pred)
        gt_parsed = parse(gt)
        return 1.0 if (pred_parsed is not None and gt_parsed is not None and verify(pred_parsed, gt_parsed)) else 0.0
    except Exception:
        return 0.0

def format_reward(completions, **kwargs):
    answer_blocks = [ANSWER_TAG_RE.findall(c_txt) for c_txt in completions]
    one_answer = [len(blocks) == 1 for blocks in answer_blocks]
    answer_txt = [blocks[0].strip() if len(blocks) == 1 else "" for blocks in answer_blocks]

    answer_matches = [bool(ANSWER_ONLY_BOXED_RE.fullmatch(a_txt)) for a_txt in answer_txt]

    answer_searches = [ANSWER_TAG_RE.search(c_txt) for c_txt in completions]
    prefixes = [c_txt[:m.start()].strip() if m else "" for c_txt, m in zip(completions, answer_searches)]
    suffixes = [c_txt[m.end():].strip() if m else "" for c_txt, m in zip(completions, answer_searches)]

    step_blocks = [STEP_BLOCK_RE.findall(prefix) for prefix in prefixes]
    step_counts = [len(blocks) for blocks in step_blocks]

    prefix_compact = [compact(prefix) for prefix in prefixes]
    blocks_compact = [compact("".join(blocks)) for blocks in step_blocks]

    return [
        1.0 if (has_one and ans_ok and suffix == "" and n_steps > 0 and p_cmp == b_cmp) else 0.0
        for has_one, ans_ok, suffix, n_steps, p_cmp, b_cmp
        in zip(one_answer, answer_matches, suffixes, step_counts, prefix_compact, blocks_compact)
    ]

def acc_reward(completions, solution, **kwargs):
    answer_blocks = [ANSWER_TAG_RE.findall(c_txt) for c_txt in completions]
    pred_txt = [last_boxes(blocks[0]) if len(blocks) == 1 else None for blocks in answer_blocks]
    gt_txt = [last_boxes(s_txt) for s_txt in solution]

    return [safe_verify(pred, gt) for pred, gt in zip(pred_txt, gt_txt)]
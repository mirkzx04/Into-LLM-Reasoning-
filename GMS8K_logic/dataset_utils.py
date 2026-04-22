import re

def split_reasoning_steps(reasoning_text: str):
    reasoning_text = reasoning_text.strip()

    # Normalize spaces and newlines
    reasoning_text = reasoning_text.replace("\r\n", "\n")
    reasoning_text = re.sub(r"\n{3,}", "\n\n", reasoning_text)

    # First split: paragraphs
    chunks = [c.strip() for c in re.split(r"\n\s*\n", reasoning_text) if c.strip()]

    steps = []
    for chunk in chunks:
        # If the paragraph is short, keep it as a single step
        if len(chunk) < 120:
            steps.append(chunk)
            continue

        # Heuristic split into sentences
        substeps = re.split(r"(?<=[.!?])\s+(?=[A-Z\\(])", chunk)
        substeps = [s.strip() for s in substeps if s.strip()]

        if substeps:
            steps.extend(substeps)
        else:
            steps.append(chunk)

    return steps


def convert_solution_to_tagged_completion(solution: str):
    solution = solution.strip()

    # If there are multiple solutions like "- OR -", keep only the first one
    solution = re.split(r"\n\s*-\s*OR\s*-\s*\n", solution, maxsplit=1)[0].strip()

    # Extract the last boxed as the final answer
    boxed_matches = re.findall(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", solution)
    final_answer = boxed_matches[-1].strip() if boxed_matches else None

    # Remove the boxed from the reasoning, leaving the rest readable
    reasoning = re.sub(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\1", solution).strip()

    steps = split_reasoning_steps(reasoning)

    tagged_steps = "\n".join(
        f"<reasoning:step>{step}</reasoning:step>"
        for step in steps
    )

    if final_answer is not None:
        completion = (
            f"{tagged_steps}\n"
            f"<answer>\\boxed{{{final_answer}}}</answer>"
        )
    else:
        # fallback if \boxed{} is not found
        completion = (
            f"{tagged_steps}\n"
            f"<answer>UNKNOWN</answer>"
        )

    return completion
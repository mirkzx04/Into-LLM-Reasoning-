import re 
from math_verify import parse, verify

def format_reward_func(completions, **kwargs) : 
    pattern = r"^<thinking:step>[\s\S]*?</thinking:step><answer>\s*\\boxed{.*?}\s*</answer>$"
    completions_content = [completion[0]["content"] for completion in completions] # Extract completion contets
    matches = [re.match(pattern, content) for content in completions_content] # Check if che completion content respects the format

    return [1.0 if match else 0.0 for match in matches]

def reward_func(completions, answer, **kwargs): 
    matches = [re.search(r"\\boxed{(.*?)}", c) for c in completions] # Extract the answer boxes
    contents = [m.group(1) if m else "" for m in matches] # Extract the number into the answer boxes

    ground_truth = [
        re.search(r"\\boxed{(.*?)}", gt).group(1).strip() if re.search(r"\\boxed{(.*?)}", gt) else gt.strip()
        for gt in answer
    ]

    return [1.0 if verify(parse(c), parse(gt)) else 0.0 for c, gt in zip(contents, ground_truth)]







    
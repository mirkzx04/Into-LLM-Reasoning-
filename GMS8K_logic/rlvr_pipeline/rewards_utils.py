import re 
from math_verify import parse, verify

def extract_answer(txt):
    """
    Search '###' into the LLM answer and take it
    """
    match = re.search(f'####\s*(.*)', txt)
    if match: 
        return match.group(1).strip()
    return None

def accuracy_reward(prompts, completions, answer, **kwargs) : 
    """
    Use math verify to check if the mathematical answer is correct.
    Assign 1.0 if the mathematical answer is correct else 0.0
    """
    rewards = []

    # Iter on batch iteration
    for i in range(len(completions)):
        ground_truth = answer[i]
        generated_txt = completions[i][0]['content'] if isinstance(completions[i], list) else completions[i]

        #  Extract the string
        pred_str = extract_answer(generated_txt)
        gt_str = extract_answer(ground_truth)

        # If the model don t place '####' into the answer the formatting is wrong 
        if pred_str is None or gt_str is None:
            rewards.append(0.0)
            continue
        
        # Math verify check if the answer and ground truth is mathematical equivalents
        is_correct = verify(parse(pred_str), parse(gt_str))
        rewards.append(1.0 if is_correct else 0.0)
    
    return rewards

def format_reward(prompt, completions, **kwargs):
    """
    Give the model a prize if the model respect formatting instructions
    """
    rewards = []
    for i in range(len(completions)):
        testo_generato = completions[i][0]['content'] if isinstance(completions[i], list) else completions[i]
        
        # If the model respect formatting instructions receives 1.0 else -1.0
        if re.search(r'####\s*[-+]?\d*\.?\d+', testo_generato):
            rewards.append(1.0)
        else:
            rewards.append(-1.0)
            
    return rewards
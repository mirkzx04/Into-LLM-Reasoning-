import os 
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from datasets import load_dataset

from analysis.extract_activation import extract_residual_out, rlvr_model, sftt_model, base_model
from GMS8K_logic.rlvr_pipeline.rewards_utils import extract_answer

platinum = load_dataset("madrylab/gsm8k-platinum", "main", split="test")

for q, a in zip(platinum['question'], platinum['answer']):
    gt = extract_answer(a)

    rlvr_answer = extract_answer(rlvr_model.generate(q))
    sftt_answer = extract_answer(sftt_model.generate(q))
    base_answer = extract_answer(base_model.generate(q))



    


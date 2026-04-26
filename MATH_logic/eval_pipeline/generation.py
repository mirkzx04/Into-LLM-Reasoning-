import torch as th

from tqdm import tqdm

from models.model import get_model, get_tokenizer
from MATH_logic.dataset_utils.dataset_splitting import build_ood_eval_dataset

device = "cuda" if th.cuda.is_available() else "cpu"
RLVR_PTH = "rlvr_model"

# Get model, tokenizer and dataset
model = get_model().eval().to(device)
tokenizer = get_tokenizer()

dataset = build_ood_eval_dataset(tokenizer, "rlvr")
dataset = dataset.add_column("id", range(len(dataset))) 

def structure_model_result(
        batch_indices, 
        effective_sample_iter, 
        decoded, 
        prompts, 
        solutions,
        result
    ):
    cursor = 0

    for local_idx, dataset_idx in enumerate(batch_indices) :
        pid = str(dataset_idx)
        answers = []

        for _ in range(effective_sample_iter):
            answers.append(decoded[cursor].strip())
            cursor += 1
        
        result[pid] = {
            "id": pid,
            "prompt": prompts[local_idx],
            "solution": solutions[local_idx],
            "answers": answers,
            }
    
    return result

def generate_answer(n_samples, sample_iter, do_sample, batch_size) : 
    # Set the number of samples that we want to process
    if n_samples is None : 
        n_samples = len(dataset)
    n_samples = min(n_samples, len(dataset))

    # Set do_sample if we want to generate more sequence 
    effective_sample_iter = sample_iter if do_sample else 1

    result = {}
    indices = list(range(n_samples))

    for start in tqdm(range(0, n_samples, batch_size), desc = "Generation") :
        batch_indices = indices[start : start + batch_size] # Set the batch indices

        # Take problem and solution from the dataset
        prompts = [dataset[i]["prompt"] for i in batch_indices]
        solutions = [dataset[i]["solution"] for i in batch_indices]

        inputs = tokenizer(
            prompts, 
            return_tensors = "pt",
            padding = True,
            truncation = True,
        ).to(model.device) # Shape : [batch_size, seq_len]
        input_len = inputs["input_ids"].shape[1]
        
        # Setting generation option
        generation_kwargs = {
            "max_new_tokens": 1024,
            "do_sample": do_sample,
            "num_return_sequences": effective_sample_iter,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "use_cache": True,
        }
        if do_sample:
            generation_kwargs.update({
                "temperature" : 0.7,
                "top_p" : 0.95
            })

        with th.inference_mode():
            outputs = model.generate(
                **inputs,
                **generation_kwargs,
            ) # Shape : [batch_size * sample_iter, seq_len]
        generated_token = outputs[:, input_len:]
        decoded = tokenizer.batch_decode(
            generated_token, 
            skip_special_tokens = True,
        )

        result = structure_model_result(batch_indices, effective_sample_iter, decoded, prompts, solutions, result)
    
    return result


from tqdm import tqdm

def map_dataset(dataset):
    print('=== LOADING DATASET ===')
    builded_dataset = {}
    for idx, example in tqdm(enumerate(dataset)):
        prompt_txt = ("Solve this mathematical problem thinking step by step."
                       "Your answer must end inserting the final result afet '####'. \n\n" 
                       f"Problem : {example['question']}"
                    )
        question_answer = example['answer']

        builded_dataset[idx] = {'prompt ' : prompt_txt, 'prompt_answer' : question_answer}

    return builded_dataset

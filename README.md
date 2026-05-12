## Abstract 
Reinforcement Learning from Verifiable Rewards (RLVR) improves problem-solving skills in LLMs. In this project, I will fine-tune open-weights models to investigate the underlying reasoning mechanisms acquired during RLVR. Training is conducted using the GSM8K dataset for mathematical reasoning and the HumanEval dataset for coding tasks. We aim to understand how RLVR enhances mathematical and algorithmic reasoning capabilities at a mechanistic level.

## Goal 
The academic community is currently debating how RLVR alters model parameters. Specifically, it remains unclear whether the model already possesses the necessary knowledge (with RLVR merely creating routing pathways to extract the correct answer) or if RLVR induces the creation of novel features.

Considering the transformer architecture as a residual stream manipulated by Attention Heads and Multi-Layer Perceptrons (MLPs), we aim to investigate the extent to which RLVR modifies internal representations versus merely acting as a behavioral wrapper.

To formalize this, we define two competing hypotheses:

*   **Steering Hypothesis (H0):** RLVR acts purely as a routing mechanism. It modifies the Attention circuits to steer pre-existing knowledge without creating new features in the MLPs.
*   **Representation Learning Hypothesis (H1):** RLVR forces the crystallization of new logical circuits, fundamentally altering the latent features encoded within the MLP layers.

To test these hypotheses, we analyze three distinct training phases:
1.   **Vanilla Phase:** The base pre-trained model before any domain-specific exposure.
2.   **Supervised Fine-Tuning (SFT) Phase:** The model trained via next-token prediction (acting as a baseline for formatting and basic knowledge).
3.   **RLVR Phase:** The model fine-tuned using RLVR on the same datasets.

By isolating these internal components, we study:
*   **Self-Attention:** To evaluate if RLVR establishes pathways toward the correct answer.
*   **MLP:** To check if RLVR alters weights or activations within the feed-forward layers, which would suggest the acquisition of new knowledge.

---

# Training Setup

## Supervised Fine-Tuning
For the SFT stage, I used the NuminaMath-CoT dataset. This provides the model with basic mathematical reasoning patterns, solution structures, and the desired response format required before applying RLVR.

## RLVR Training
I constructed a mixed mathematical dataset using GSM8K, MATH-Lighteval (filtered by level), and DAPO-Math-17k. While SFT teaches the model to imitate traces, RLVR optimizes the model toward solution trajectories that maximize verifiable correctness.

## RLVR Configuration
Using the Hugging Face `trl` library, specifically `GRPOConfig` and `GRPOTrainer`:
*   **Learning Rate:** 2e-6
*   **Max Completion Length:** 2000
*   **Loss Type:** DAPO (chosen for its effectiveness with variable completion lengths).
*   **Infrastructure:** DeepSpeed for memory efficiency and vLLM for fast generation sampling.

---

# Experiments

## Dataset Building
We constructed a dataset of internal activations extracted from the BASE, SFT, and RLVR versions of the model. To ensure comparability, we use the RLVR model to generate a completion, then feed that exact sequence into all three models to extract activations at the same textual positions.

The sequence consists of the prompt and the completion. For every model and selected layer, we save:
*   **Pre-residual:** The input stream entering the layer.
*   **MLP output:** The specific contribution of the MLP block.
*   **Attention output:** The specific contribution of the Self-Attention block.

In a standard transformer layer, the "middle" residual state is the sum of the **pre-residual** and the **attention output**. The final "post-residual" state is the sum of that **middle state** and the **MLP output**.

### Activation Dataset Reproducibility

To reproduce the activation dataset, run `get_activation_dataset` from `experiments/experiments_main.py`.

This function builds a shared token cache using a generator model, then replays the same prompt-completion sequences through all model variants to extract comparable activations.

### Main inputs

- `gen_model`: model used to generate the completion.
- `gen_tokenizer`: tokenizer associated with `gen_model`.
- `gen_dataset`: dataset used for generation.
- `model_desc`: list of `(model_path, model_name)` pairs for the models to compare.
- `save_path`: directory where the activation dataset will be saved.
- `generator_name`: identifier of the model used for generation.
- `ood_dataset_name`: identifier of the evaluation dataset.
- `max_new_tokens`: maximum completion length used during generation.

### Saved files

Running the pipeline creates:

- `save_path/<dataset_name>.h5`: activation dataset in HDF5 format
- `save_path/<dataset_name>_metadata.pt`: lightweight metadata for the activation dataset
- `save_path/tokens_cache/<token_cache_prefix>.pt`: full token cache
- `save_path/tokens_cache/<token_cache_prefix>.jsonl`: JSONL export of the token cache

### Extraction procedure

1. The generator model produces one completion for each prompt in `gen_dataset`.
2. Prompt tokens and generated completion tokens are saved in a token cache.
3. The same full sequence (`prompt + completion`) is replayed through every model listed in `model_desc`.
4. Activations are extracted at the same token positions for all compared models.

### Current activation setup

At the moment, activations are extracted for:
- the first layer
- the middle layer
- the last layer

For each selected layer, the following activations are stored:
- `resid_pre_act`
- `attn_out_act`
- `mlp_out_act`

The saved token positions cover the full sequence:
- all prompt tokens
- all completion tokens

### HDF5 structure

```text
<dataset_name>.h5
│
├── <model_name_1>/
│   ├── index/
│   │   ├── sample_id
│   │   ├── start
│   │   ├── end
│   │   ├── prompt_len
│   │   ├── completion_len
│   │   └── total_len
│   │
│   ├── layer_00/
│   │   ├── mlp_out_act      # [total_tokens, d_model]
│   │   ├── attn_out_act     # [total_tokens, d_model]
│   │   └── resid_pre_act    # [total_tokens, d_model]
│   │
│   ├── layer_XX/
│   │   ├── mlp_out_act
│   │   ├── attn_out_act
│   │   └── resid_pre_act
│   │
│   └── layer_YY/
│       ├── mlp_out_act
│       ├── attn_out_act
│       └── resid_pre_act
│
└── <model_name_2>/
    ...
```

Here, `start` and `end` define the row span of each sample inside the flattened activation matrices.

### Notes
* The exact group names at the top level of the HDF5 file depend on the `model_name` values passed in `model_desc`.
* If you want to change which layers are extracted, modify the layer-selection logic in `extract_activation.py`.
* The token cache `.pt` file stores the generated tokenized dataset, while the `*_metadata.pt` file stores only lightweight metadata about the activation dataset.

## 1. CKA
The conducted tests have shown how the MLP is the one to undergo greater geometric transformations. Their intensity is not very high, therefore it is not a global reorganization of the representations, however the differences increase in the deep layers and especially in the outputs of the MLPs linked to the deepest positions of the completions.

Going up to 95% of the completion tokens we notice how the MLP undergoes stronger geometric transformations compared to the Attention:

**CKA | MLP ACT OUT | POSITION 0.95**

| Layer | base-sftt | base-rlvr | sftt-rlvr |
| --- | --- | --- | --- |
| layer_00 | 0.9999 | 0.9995 | 0.9995 |
| layer_14 | 0.9582 | 0.9135 | 0.9478 |
| layer_27 | 0.7016 | 0.6826 | 0.8696 |
| mean | 0.8866 | 0.8652 | 0.9390 |

**CKA | ATTN ACT OUT | POSITION 0.95**

| Layer | base-sftt | base-rlvr | sftt-rlvr |
| --- | --- | --- | --- |
| layer_00 | 1.0000 | 0.9999 | 0.9999 |
| layer_14 | 0.9579 | 0.9430 | 0.9673 |
| layer_27 | 0.8828 | 0.9153 | 0.9041 |
| mean | 0.9469 | 0.9527 | 0.9571 |

The transformations are greater in the last layers, indicating that the representations have undergone greater transformations, this suggests that the fine-tuning acts more on the deep MLPs of the model. The sftt-rlvr models remain quite close, indicating how RLVR preserves the representational structure of SFT.

As seen in [Filtering with Self-Attention and Storing with MLP One Layer Transformers Can Provably Acquire and Extract Knowledge](https://arxiv.org/pdf/2508.00901v3a) the MLP modules absorb and extract knowledge, the geometry changes in the MLP in the deep layers could indicate a reorganization, more or less intense, of the knowledge learned in SFT, a reorganization does not imply the addition of new knowledge, the representations of some concepts or patterns might have been learned during SFT and refined (or made more decodable) by RLVR.

## 2. Linear Probing & Causal Intervention
### 2.1 Native-Logit Lens 
#### 2.1.1 Component-Level Logit Divergence: Top-K Overlap and Symmetric KL
We use Top-K Jaccard to quantify the overlap of the top-k predicted tokens across the models, we also use Symmetric KL Divergence to quantify how much the distributions on the logits differ. In this section we will look at the delta of Symmetric KL and the delta of Top-k Jaccard.

**Delta Symmetric KL**
If the delta is greater than zero it means that the MLP makes the distributions more different, conversely it is the attention making them more different if the delta is less than zero.

![](experiments/Logit_Lens/logit_lens_img/pairwise_dkl_delta/native/delta_pairwise_DKL.png)

Observing the graphs we notice how the scores greater than zero are mainly in the intermediate layer, in particular for positions 0.50 and 0.95, and in the initial layer of position 0.95. These two patterns confirm that the MLP shifts the distribution mainly in the intermediate layer; being the MLP associated with the storage of knowledge and its extraction, we could think that the intermediate layer is the point where the knowledge is extracted the most. In the last position we notice how the initial layer is the one with the highest scores, as the layers advance the scores converge to zero, looking however at the uncertainty band the estimate is less stable across samples: some completions show a much more divergent MLP contribution, others show a more divergent or almost null attention contribution.

**Delta Top-K Jaccard**
When the delta is less than zero it means that the Attention is more prone to return a greater overlap of similar tokens compared to what the MLP returns.

(base, rlvr) is the graph that always remains mostly below zero in the last layer, the deepest layer; as already mentioned before, it is precisely in the deepest layers where the major geometry change occurs, this indicates that the MLP incentivizes different tokens between the two models in that layer. In the paper [Towards a Mechanistic Undestanding of LRM - A survey of training and inference.](https://arxiv.org/pdf/2601.19928) it is shown how RLVR models face two training stages, in the second stage they learn to use the most optimal tokens. Observing the graphs it is plausible to think that it could be precisely the MLP that incentivizes the use of the most optimal tokens.

![](experiments/Logit_Lens/logit_lens_img/pairwise_jaccard_delta/native/delta_jaccard.png)

#### 2.1.2 Attention vs MLP Target Log-Probability Contribution
Generally, the attention is the module that shifts the probability distribution towards the target token; this is especially true for the initial layers. By zooming in on the intermediate and final layers, we can notice how in the first position in the sftt model, it is the MLP that has a greater effect on the distribution, shifting it towards the target token.

|  |  |
| :---: | :---: |
|![](experiments/Logit_Lens/logit_lens_img/component_logprob/native/component_preference.png)       | ![](experiments/Logit_Lens/logit_lens_img/component_logprob/native/component_preference__without_layer0.png)

In position 0.50 in the sftt model, the attention returns to dominate the Log-Probability distribution, while in the rlvr model, the effect of the attention module weakens. Position 0.95 is the one that shows a clearer hierarchy: the sftt model confirms itself as the one where the attention has a lighter effect on the distribution, but its uncertainty bands suggest that the MLP module has a greater weight on the prediction of the target token in some samples.

### Interpretation

The MLP module is the one more prone to shift the probabilistic mass of the output vocabulary. As seen from **Symmetric KL**, the distributions with the addition of the MLP to the residual signal tend to shift the probability mass, especially between the base and rlvr models. However, the MLP also has a second effect: introducing new tokens into the pool of tokens chosen in output by the model. This effect is not constant across all layers nor in all positions; it tends to be more marked in positions 0.10 and 0.50. This could indicate how the MLP changes the semantics of the internal reasoning; indeed, the MLP could be the module that inserts the optimal reasoning tokens within the pool of experts.

Despite the MLP being the one to shift the probability mass and introduce some new tokens, it is the Attention module that points to the output token. The effect of the attention on the probability of the output token is higher than that of the MLP, especially in the initial layers.

These two analyses indicate that the MLP is not the module choosing the output token; its effect is more tied to the internal representations of the model and to the probabilistic mass assigned by the model in the distribution over the output vocabulary. The change in geometries observed in CKA might have led precisely to this effect. The MLP might contribute more significantly to assigning a probability distribution to the output tokens. After the SFT and RLVR trainings, the MLP might shift the distribution onto new tokens, which could coincide with the optimal tokens that RLVR should prefer.

### 2.2 Linear Probing Answer
We take the activation vector for a reasoning token and pass it through a linear layer with a Softmax function to calculate the probability of specific classes among A, B, C or D.
### 2.3 Activation Patching
To establish causality, we inject specific activations from the RLVR model into the SFT model. This proves whether a specific MLP or Attention layer holds the critical features for successful reasoning.

## 3. Weight Distance & Spectral Analysis
To quantify parameter updates, we calculate the distance between the weight matrices of the models (using the L2 norm). 

To look deeper, we use **Singular Value Decomposition (SVD)** on the matrix representing the difference between weights. If RLVR is just a steering mechanism, these weight updates should show a "low rank"—meaning the changes are concentrated in a few specific routing heads rather than being spread across the MLP knowledge layers.

*For a more in-depth analysis of the experiments, you can refer to::* [experiments](doc/experiments.md)
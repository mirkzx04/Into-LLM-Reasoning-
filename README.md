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

## Component-Level Representation Comparison Summary

This analysis compares internal representations across three model training stages—**BASE**, **SFT**, and **RLVR**—to determine if fine-tuning causes large global changes in the geometry of Attention and MLP outputs. We compute **linear Centered Kernel Alignment (CKA)** between pairs of model-layer representations using two component outputs: `attn_out_act` and `mlp_out_act`. 

Activations of shape `[L, seq_len, d_model]` are sliced into `[N, d_model]` matrices using two distinct strategies.

---

### 1. Last Input Token 
We study the geometry of the activations at the position of the last input token, that is, in the state that conditions the distribution of the first token generated by the model.

#### MLP 
The geometry remains globally very similar across the models, but the reduction in CKA in the final layers indicates that the differences introduced by fine-tuning are mainly concentrated in the deep layers.

<img src = "experiments/CKA/cka_img/mlp/mlp_out_last_inp_tok.png" width="750">

Observing the table of numerical **CKA** values we have confirmation: The geometries change more in the deeper layers.
| Layer | base-sftt | base-rlvr | sftt-rlvr |
| --- | ---: | ---: | ---: |
| layer_00 | 1.0000 | 0.9999 | 0.9999 |
| layer_14 | 0.9965 | 0.9876 | 0.9908 |
| layer_27 | 0.9172 | 0.9511 | 0.9323 |
| mean | 0.9712 | 0.9795 | 0.9743 |

`layer_27` is the one showing the largest change. By studying these changes, we can notice that the geometries of the `sftt-rlvr` comparison are more similar compared to the geometries of the `base-sftt` comparison. As already shown in [Filtering with Self-Attention and Storing with MLP One Layer Transformers Can Provably Acquire and Extract Knowledge](https://arxiv.org/pdf/2508.00901v3a), supervised fine-tuning refines the knowledge embedded in the MLPs of the base model, trained on next-token-prediction (NTP). This is also consistent with the **CKA** values observed in the table.

The fact that base-rlvr is more similar than base-sftt suggests that, at least on the last input token, RLVR does not necessarily amplify the geometric shift introduced by SFT. A possible interpretation is that RLVR selects or reuses some representations already present in the base model, instead of simply continuing in the geometric direction introduced by SFT.

In subsequent tests, we compare these results with the attention activations, to verify if the same pattern is also observed in the routing modules.

### 2. Predictive completion mean
We want to study the intensity of the geometric change on the positions that predict the tokens of the generated completion. This test aggregates the activations along the reasoning sequence, allowing us to compare the average geometry used by the models during generation.

We take the token sequence corresponding to all generated tokens and calculate the mean along the token axis: 
```python 
pred_start = prompt_len - 1
pred_end = prompt_len + completion_len - 1
act = act[:, pred_start:pred_end, :].mean(dim=1)
```

Thanks to this structure, we can study how the geometries of the models vary on activations of tokens predicted during the reasoning steps. Observing the heat-maps, we can notice a fairly repetitive pattern: The geometries between the same layers of different models do not change excessively.

#### MLP
In the MLP we have a higher intensity, mostly isolated on the last layer; in fact, layers `layer_00` and `layer_14` have very similar if not identical geometries. Observing `layer_27`, we notice a more marked reduction in CKA, hence a greater geometric difference between the models.

<img src = "experiments/CKA/cka_img/mlp/mlp_out_pred_comp_mean.png" width="750">

Observing the table of complete **CKA** values also confirms this pattern: 
| Layer | base-sftt | base-rlvr | sftt-rlvr |
| --- | ---: | ---: | ---: |
| layer_00 | 1.0000 | 0.9998 | 0.9998 |
| layer_14 | 0.9890 | 0.9885 | 0.9871 |
| layer_27 | 0.9165 | 0.8945 | 0.9561 |
| mean | 0.9685 | 0.9609 | 0.9810 |

The deeper layers are the ones that have a greater variation in geometry. We can also notice how the geometries between `sftt-rlvr` are much more similar compared to `base-rlvr`, contrary to what was seen on the activations of the last input token.

During the generation of the completion, the geometry of rlvr is closer to that of sftt than to that of the base model. This suggests that RLVR preserves a significant part of the representational structure introduced by SFT during the generation of the reasoning.

#### Attention 
The attention shows geometries that are much more similar to each other compared to those of the MLP; from the heat-map, no easily noticeable changes are highlighted.

<img src = "experiments/CKA/cka_img/act/attn_out_pred_comp_mean.png" width="750">

Looking at the table of **CKA** values we can notice two interesting patterns: 
- The attention geometries of the `base` model and the `sftt` model are closer compared to the MLP geometries.
- The attention geometries of the `sftt` model and the `rlvr` model are closer compared to the MLP geometries.

| Layer | base-sftt | base-rlvr | sftt-rlvr |
| --- | ---: | ---: | ---: |
| layer_00 | 1.0000 | 0.9999 | 1.0000 |
| layer_14 | 0.9982 | 0.9915 | 0.9919 |
| layer_27 | 0.9663 | 0.9609 | 0.9736 |
| mean | 0.9882 | 0.9841 | 0.9885 |

[A Mathematical Framework for transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html#splitting-attention-head-terms-into-circuits) Since we measure `attn_out` here, the result does not directly identify changes in QK routing patterns. Rather, it indicates that the output written by the attention into the residual stream, OV, remains geometrically more stable compared to the MLP outputs. In the studied setup, the greater geometric stability of `attn_out` suggests that the differences introduced by SFT and RLVR are more visible in the MLPs than in the attention output. This does not rule out a role for attention, but indicates that the change does not clearly emerge from this specific measure.

The activations in the comparisons are very similar; this does not indicate that the attention would have predicted very similar token distributions, but by observing the geometric changes we can highlight that `sftt` and `rlvr` had a greater impact on the MLPs. From these two tables, we can highlight that, although marginal, there are changes in the geometries of the different components, with an MLP that receives a greater intensity of change, on average, compared to the attention.

### 3. Position Completion
In this way, we can study how the geometry of the activations changes in different normalized positions of the completion, verifying if the differences between models increase as the reasoning proceeds.
```python 
rel_idx = int(round((completion_len - 1) * normalized_pos))
rel_idx = _clamp(rel_idx, 0, completion_len - 1)
abs_idx = completion_start + rel_idx
return act[:, abs_idx, :]  
```

This way we can study how the activation geometries of the three models change across the token length of the completion; it is useful to see with what geometry the models represent the reasoning.

#### MLP 
Observing the heat-map, we can see that the geometries between the layers remain very similar: 
- `layer_00`: The geometries remain very similar. 
- `layer_14`: Here too, the geometries remain very similar. 
- `layer_27`: Here the greatest difference in geometries is perceived, albeit marginal.

<img src = "experiments/CKA/cka_img/mlp/mlp_norm_pos_025.png" width="750">

Observing the table that returns the full **CKA** score: 
| Layer | base-sftt | base-rlvr | sftt-rlvr |
| --- | ---: | ---: | ---: |
| layer_00 | 1.0000 | 0.9996 | 0.9996 |
| layer_14 | 0.9807 | 0.9688 | 0.9718 |
| layer_27 | 0.9632 | 0.9301 | 0.9490 |
| mean | 0.9813 | 0.9662 | 0.9735 |


The previously seen pattern is confirmed again; the MLP blocks are the ones that have a greater intensity of geometric change compared to the attention. The MLP activations of the first `25%` of the generated tokens show how the geometries are close, indicating that to generate the first `25%` of the tokens the models do not drift excessively from each other.

Reaching up to `75%` of the generated tokens, we get closer to the previously seen pattern.
| Layer | base-sftt | base-rlvr | sftt-rlvr |
| --- | ---: | ---: | ---: |
| layer_00 | 1.0000 | 0.9995 | 0.9995 |
| layer_14 | 0.9873 | 0.9730 | 0.9757 |
| layer_27 | 0.9788 | 0.9249 | 0.9453 |
| mean | 0.9887 | 0.9658 | 0.9735 |

At 75% of the completion, the largest difference still emerges in the `base-rlvr` comparison, while `sftt-rlvr` remains closer. This suggests that, in the intermediate phases of the completion, RLVR separates more from the base model while maintaining a geometry closer to SFT.

Reaching up to `95%` of the generated tokens, the pattern is confirmed: 
| Layer | base-sftt | base-rlvr | sftt-rlvr |
| --- | ---: | ---: | ---: |
| layer_00 | 0.9999 | 0.9995 | 0.9995 |
| layer_14 | 0.9582 | 0.9135 | 0.9478 |
| layer_27 | 0.7016 | 0.6826 | 0.8696 |
| mean | 0.8866 | 0.8652 | 0.9390 |

This time we have much more different geometries among them; the pattern is confirmed again to be greater in the deep layers.

The geometries of the last layer shift significantly, especially in `base-rlvr`, indicating that the activations have changed between the two. This sharp drop in CKA compared to the base model suggests that, towards the end of the completion, SFT and RLVR traverse representational regions more distant from those of the base model. A possible hypothesis is that the deep MLP layers encode or combine features more specific to mathematical reasoning.
The most marginal difference remains between `sftt-rlvr`, indicating that between the two versions of the model the activation geometries are much more similar. This could be due to the fact that the MLP features of `rlvr` manage to activate more compared to the features of `sftt`, precisely because RLVR might have made the use of some representations already acquired during SFT more systematic.

#### Attention 
Also for the attention, the geometries remain very similar. Compared to the MLPs, the changes are more contained: in the heat-map, the differences visible in the MLPs are almost completely absent.

<img src = "experiments/CKA/cka_img/act/attn_out_position_normalized_completion_025.png" width="750">

Observing the table of activations up to `25%` of the generated tokens, we can notice a much more contained change in geometries compared to the MLPs. Moreover, the scores remain highly aligned with those seen in the previous test.
| Layer | base-sftt | base-rlvr | sftt-rlvr |
| --- | ---: | ---: | ---: |
| layer_00 | 1.0000 | 0.9999 | 0.9999 |
| layer_14 | 0.9893 | 0.9797 | 0.9805 |
| layer_27 | 0.9859 | 0.9781 | 0.9810 |
| mean | 0.9917 | 0.9859 | 0.9871 |

Going up to `95%` of the generated tokens, we notice the same pattern noted in the MLP, the geometries differentiate more: 
| Layer | base-sftt | base-rlvr | sftt-rlvr |
| --- | ---: | ---: | ---: |
| layer_00 | 1.0000 | 0.9999 | 0.9999 |
| layer_14 | 0.9579 | 0.9430 | 0.9673 |
| layer_27 | 0.8828 | 0.9153 | 0.9041 |
| mean | 0.9469 | 0.9527 | 0.9571 |

The change in the attention activations with greater reasoning depth could be due to the fact that the attention has learned new patterns among tokens; with this, it might have learned to pay more attention to tokens that it previously considered uninformative.
Unlike the MLPs, in the attention the most distant comparison is not always base-rlvr; at layer_27, the lowest value is base-sftt. This suggests that the changes in the attention are weaker and less structured compared to those observed in the MLPs.

Since this test measures `attn_out` and not directly the attention probabilities, we cannot conclude that the model has learned new routing patterns. We can, however, observe that towards the end of the completion, the attention output also shows a reduction in geometric similarity between models. 

Recent literature on RLVR highlights that training with verifiable rewards can be associated with entropy reduction dynamics and a greater concentration of the distribution on the tokens. However, from the CKA results on `attn_out` alone, we cannot attribute this dynamic to the attention. To verify this, it will be necessary to compare logit entropy, Logit Lens/Logi-Lens and, separately, causal interventions on MLP and attention [Towards a Mechanistic Understanding of LRM - A survey of training and inference.](https://arxiv.org/pdf/2601.19928).

The MLP shows greater geometric variation compared to the attention, because the CKA values are lower.
| Activation module | base-sftt | base-rlvr | sftt-rlvr |
|-------------------|------------|-----------|----------|
| Attention         | 0.9469    | 0.9527     | 0.9571   |
| **MLP**           | **0.8866** | **0.8652** | **0.9390** |

### Interpretation 
The CKA results show that SFT and RLVR do not produce a global reorganization of the representations, given that the geometric similarity remains high in most layers and positions. However, the differences increase in the deep layers, especially in the MLP outputs and in the final positions of the completion. This suggests that fine-tuning acts more heavily on the deep MLP components during the generation of the reasoning, while the attention output remains more stable. Furthermore, the greater closeness between sftt and rlvr during the completion indicates that RLVR seems to preserve part of the representational structure introduced by SFT, rather than producing a completely new geometry.

## 2. Linear Probing & Causal Intervention
We train linear classifiers on the hidden states to predict correct intermediate reasoning steps. 
*   **Linear Probing Answer:** We take the activation vector for a reasoning token and pass it through a linear layer with a Softmax function to calculate the probability of specific classes among A, B, C or D.
*   **Activation Patching:** To establish causality, we inject specific activations from the RLVR model into the SFT model. This proves whether a specific MLP or Attention layer holds the critical features for successful reasoning.

## 3. Weight Distance & Spectral Analysis
To quantify parameter updates, we calculate the distance between the weight matrices of the models (using the L2 norm). 

To look deeper, we use **Singular Value Decomposition (SVD)** on the matrix representing the difference between weights. If RLVR is just a steering mechanism, these weight updates should show a "low rank"—meaning the changes are concentrated in a few specific routing heads rather than being spread across the MLP knowledge layers.

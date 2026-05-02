# Abstract

*Reinforcement Learning from Verifiable Rewards* (RLVR) improves problem-solvin skills in LLMs. In this project I’ll fine tune open weights models to investigate the underlying reasoning mechanism acquired during *RLVR.* Training will be conducted using the *GSM8K* for mathematical reasoning and the *HumanEval* dataset for coding task. We aim to understand how *RLVR* enhances mathematical and algorithmic reasoning capabilities at mechanist level.

# Goal

The academic community is currently debating how *RLVR* alters model parameters. Specifically it remain unclear whather the model already possesses the necessary knowledge with *RLVR* merely creating routing pathways to extract the correct answer or, if *RLVR* induces the creation of novel features.

Considering the transformers architecture as a residual stream manipulated by Attention Heads and MLPs, we aim to investigate the extent to which *RLVR* modifies internal representation versus merely acting as a behavioral wrapper.

To formalize this, we define two competing hypotheses:

- $H_0$ **(Steering Hypothesis):** RLVR acts purely as a routing mechanism. It modifies the Attention circuits to steer pre-existing knowledge without creating new features in the MLPs.
- $H_1$ **(Representation Learning Hypothesis):** RLVR forces the crystallization of new logical circuits, fundamentally altering the latent features encoded within the MLP layers

To test these hypotheses, we will analyze three distinct training phases of the same model:

1. **Vanilla Phase**: The base pre-trained model before any domain-specific exposure.
2. **Supervised Fine-Tuning (SFT) Phase**: The model trained via next-token prediction on our datasets without *RLVR* (acting as a baseline for formatting and basic knowledge acquisition).
3. **RLVR Phase**: The model fine-tuned using RLVR on the same datasets

By isolating these internal components, we will study:

- **Self-Attention**: To evaluate if *RLVR* merely establishes pathways toward the correct answer.
- **MLP**: To check if *RLVR* alters weights or activations within the feed-forward layers, which would strongly suggest the acquisition of new knowledge/features.

# Training Setup

## Supervised Fine-Tuning
For the SFT stage, I used the NuminaMath-CoT dataset. This choice is motivated by recent work suggesting that a strong supervised fine-tuning phase is important before applying RLVR, since it provides the model with basic mathematical reasoning patterns, solution structure, and the desired response format.

The SFT model achieves a high `mean_token_accuracy`, which indicates that it has learned to imitate the supervised solutions effectively, including the expected reasoning format and mathematical answer structure. However, this metric should mainly be interpreted as a measure of supervised imitation quality, rather than as direct evidence of robust mathematical generalization.

## RLVR Training
For the RLVR stage, I constructed a mixed mathematical dataset composed of:

1. GSM8K  
2. MATH-Lighteval, filtered by specific levels and topics  
3. DAPO-Math-17k, filtered for sample number

The goal of this dataset mixture is to expose the model to heterogeneous mathematical problems, ranging from grade-school arithmetic reasoning to more diverse competition-style problems. While SFT teaches the model to imitate supervised reasoning traces, RLVR is used to further optimize the model toward solution trajectories that maximize verifiable correctness.

In this sense, RLVR is not assumed to create OOD generalization by itself. Instead, the training setup is designed to test whether reward-based optimization over a heterogeneous dataset can improve both in-distribution accuracy and out-of-distribution generalization.

## RLVR Configuration
For RLVR training, I used the Hugging Face `trl` library, specifically `GRPOConfig` and `GRPOTrainer`.

The main hyperparameters were:

- learning rate: `2e-6`
- maximum completion length: `2000`
- loss type: `dapo`

I used the DAPO loss because the model does not produce extremely long chain-of-thought completions and because completion lengths are relatively variable across samples.

The training setup also used DeepSpeed for memory-efficient distributed optimization and vLLM for faster generation during RLVR sampling.

## Reward Function

The reward function is inspired by the DeepSeek-R1 training setup and combines an accuracy reward with a format reward.

The accuracy reward assigns:

- `+1.0` if the final answer is correct
- `0.0` otherwise

The format reward assigns:

- `+1.0` if the completion follows the required response format
- `0.0` otherwise

Since the model already learns the target format during SFT, the format reward is given a small weight of `0.01`. The main objective of RLVR is therefore to maximize mathematical correctness rather than simply reinforce formatting behavior.

The total reward can be written as:
$R = R_{\text{accuracy}} + 0.01 R_{\text{format}}$ where $R_{\text{accuracy}}$ is the dominant component.

![alt text](image.png)

1. **Generated** : 

# Experiments List

1. **Component-Level Representation Comparison**
    
    For each problem type, we will extract the hidden states across the three model versions ($h^{v}_l, h^{SFT}_l$, $h^{RLVR}_l$). Instead of solely comparing global states, we will isolate the outputs of the Attention $z_l$ and MLP $m_l$ blocks. We will use Centered Kernel Alignment to measure representational similarity, identifying which specific component diverges most heavily after *RLVR*.
    
2. **Linear Probing & Causal Intervention**
    
    We will train linear classifiers on the hidden states to predict correct intermediate reasoning steps, paired with Logit Lens analysis to map intermediate representations to the vocabulary space. To establish causality, we will employ **Activation Patching** (Causal Tracing): by injecting specific activations from the *RLVR* model into the SFT model during inference, we aim to definitively prove whether a specific MLP or Attention layer holds the critical reasoning features.
    
3. **Weight Distance & Spectral Analysis**
    
    To quantify parameter updates, we will compute the $L_2$ norm of the weight differences: $||W_{RLVR} - W_{SFT}||$ and $||W_{RLVR} - W_v||$. To overcome the limitations of $L_2$ orms in overparameterized networks, we will perform Singular Value Decomposition (SVD) on the weight difference matrix $\Delta W$. If *RLVR* primarily acts as a steering mechanism, the weight updates should exhibit a low rank and concentrate in specific routing heads rather than MLP layers.



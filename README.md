# Learning to Activate Attention

## Conditional Attention Capacity Scaling, Adaptive Depth, and Length Extrapolation

**Research area:** Efficient Attention, Conditional Computation, Long Context, ML Systems

**Project scope:** Initial architecture study and controlled small scale experiments

## Abstract

MoE demonstrates that model parameter capacity can grow while the computation activated for each token remains constrained. This project investigates whether a similar scaling principle can be developed for Attention itself.

We propose a provisional framework called **Activation Attention (AA)**, in which a model contains multiple parameterized Attention paths and learns to select a subset according to the current input. The initial design uses shared Key and Value representations with independently parameterized Query and Output projections, allowing total Attention capacity to increase while keeping the number of activated paths fixed.

The primary research question is whether increasing total Attention capacity improves language modeling and information processing under a matched active computation budget. Subsequent experiments will introduce adaptive Attention depth and investigate how capacity allocation, activation depth, and context length interact.

The project emphasizes controlled experiments, explicit computational constraints, and theoretical analysis of model capacity. Its longer term direction is to combine conditional Attention computation with learned memory management for agents operating over continuously growing histories.

## 1. Motivation

Existing efficient Attention research primarily addresses how information interaction is performed.

Full Attention supports global token interactions but has quadratic sequence length complexity in its attention computation. FlashAttention improves the memory efficiency of exact Attention through IO aware algorithms. Linear Attention changes the computational formulation, while Sparse Attention reduces the interactions that are actually computed. Hybrid architectures combine multiple information processing mechanisms.

A complementary question concerns how the Attention parameter capacity itself should scale.

MoE separates total parameter capacity from active computation by selecting a subset of experts. A similar principle may apply to Attention: a model could contain a growing collection of parameterized Attention paths while activating only a limited subset for each token.

Relevant precedents include *Mixture of Attention Heads* (EMNLP 2022), which studies token conditional selection of parameterized Attention heads, and *Native Sparse Attention* (2025), which integrates trainable hierarchical sparsity into Attention. These works establish important foundations for conditional Attention computation and trainable information access.

The proposed contribution is to investigate **the relationship between total Attention capacity, activated computation, and model quality**, with adaptive depth and length extrapolation as subsequent research questions.

## 2. Research Questions

**RQ1: Attention capacity scaling.** Can increasing the number of available Attention paths improve model quality while keeping the number of activated paths and active computation approximately constant?

**RQ2: Adaptive depth.** Can a model learn to allocate different amounts of serial Attention computation to different tokens, improving the quality versus compute tradeoff?

**RQ3: Length extrapolation.** How do conditional activation and adaptive depth affect model performance beyond the training context length? Does the routing policy remain stable as the context grows?

The primary hypothesis is that conditional Attention capacity can provide useful additional representational flexibility under a fixed active computation budget. The hypothesis is falsifiable: additional paths may learn redundant functions, fail to specialize, or provide gains too small to justify routing overhead.

## 3. Initial Architecture

### 3.1 Shared KV Attention Expert Bank

Let a Transformer layer have \(E\) candidate Attention paths. Each path has its own Query and Output projections, while all paths share Key and Value projections.

For hidden states \(H\):

\[
K=HW_K,\qquad V=HW_V
\]

For expert \(e\):

\[
Q^{(e)}=HW_Q^{(e)}
\]

\[
O^{(e)}
=
\operatorname{Attn}(Q^{(e)},K,V)W_O^{(e)}
\]

A router produces scores for each token and selects the Top-\(K\) paths:

\[
\mathcal S_i=\operatorname{TopK}(g_\phi(h_i))
\]

The output becomes:

\[
h_i'=h_i+
\sum_{e\in\mathcal S_i}
\alpha_{i,e} O_i^{(e)}
\]

where \(\alpha_{i,e}\) denotes normalized routing weights.

This design allows the number of candidate paths to increase without requiring every token to execute every path. Sharing KV also prevents KV Cache size from scaling directly with the number of experts.

The initial design deliberately limits expert heterogeneity. It tests conditional parameter capacity before introducing separate KV representations or heterogeneous Attention operators.

### 3.2 Capacity and Compute

Define:

\[
P_{\mathrm{total}}
=
P_{\mathrm{shared}}+E P_{\mathrm{expert}}
\]

\[
P_{\mathrm{active}}
\approx
P_{\mathrm{shared}}+K P_{\mathrm{expert}}
\]

For equal sized experts, increasing \(E\) expands total capacity while fixed \(K\) constrains the number of active paths.

These parameter quantities are only approximate indicators of compute. The actual cost must include routing, shared projections, attention interactions, data movement, and execution overhead.

Importantly, fixed \(K\) does not remove the quadratic sequence length term of Full Attention. Activation Attention addresses conditional parameter capacity; reductions in long sequence interaction cost require additional mechanisms.

## 4. Training Objective

### 4.1 Phase One: Fixed Activation Budget

Use standard autoregressive next token prediction:

\[
\mathcal L_{\mathrm{LM}}
=
-\mathbb E_x
\sum_t \log p_\theta(x_t\mid x_{<t})
\]

Each token activates exactly \(K\) paths. The router receives task supervision through the language modeling objective, with no manually specified expert labels.

No compute budget penalty is required initially because the number of activated paths is structurally fixed. Monitor expert utilization, routing entropy, output similarity, and leave one expert out ablations to determine whether the paths develop distinct functions.

For optimization, compare a differentiable routing approximation with a hard Top-\(K\) forward pass using an appropriate gradient estimator. The final evaluation must use hard routing and actual conditional execution.

### 4.2 Phase Two: Variable Activation and Depth

When activation count or depth becomes dynamic, introduce a compute constrained objective:

\[
\mathcal L
=
\mathcal L_{\mathrm{LM}}
+
\lambda_C
\max(0,\mathbb E[C]-B)
\]

Here \(C\) represents the measured or estimated computation per input and \(B\) is the target budget.

Load balancing and diversity losses remain optional. They should be introduced only when experiments identify expert collapse, serious load imbalance, or redundant functions.

## 5. Adaptive Attention Depth

Let \(L\) denote maximum network depth. Define the execution gate for token \(i\) at layer \(l\) as:

\[
z_{i,l}\in\{0,1\}
\]

The effective Attention depth is:

\[
D_i=\sum_{l=1}^{L}z_{i,l}
\]

The second stage introduces learned gates that allow tokens to skip selected Attention updates.

The objective is to investigate whether dynamic depth improves the quality versus compute tradeoff under a fixed budget.

Three questions will be tested:

1. Do tokens and tasks develop different effective Attention depths?
2. Does dynamic depth outperform fixed depth at matched active FLOPs?
3. Does the depth distribution change systematically as context length increases?

The analysis must account for shared Key and Value computation needed by later tokens. Skipping an Attention update does not automatically eliminate every operation associated with that token. Savings must be verified through actual execution measurements.

## 6. Experimental Plan

### Stage A: Small Scale Pretraining

Train a small causal Transformer, with an initial target of approximately 100M to 300M parameters subject to available compute.

Keep maximum depth fixed. Vary the number of experts \(E\) and active paths \(K\). Compare:

1. Standard Multi Head Attention.
2. Activation Attention with increasing \(E\) and fixed \(K\).
3. A token conditional Attention baseline inspired by prior Mixture of Attention Heads work.

Control training data, token budget, positional encoding, and model configuration as closely as possible.

Compare models under matched active FLOPs and report total parameters separately. Include total parameter matched experiments where feasible.

### Stage B: Adaptive Depth

Starting from the best validated Activation Attention configuration, introduce token level depth gating.

Compare fixed depth and adaptive depth under matched active computation budgets. Evaluate training stability, routing behavior, validation loss, throughput, and actual saved operations.

### Stage C: Length Extrapolation

Train at a fixed context length and evaluate at that length and progressively longer lengths, for example \(n_{\mathrm{train}}\), \(2n_{\mathrm{train}}\), and \(4n_{\mathrm{train}}\), subject to model and hardware limits.

Measure:

- Language modeling validation loss by context length.
- Long distance retrieval and multi hop reasoning.
- Performance under varying evidence positions and distractor density.
- Expert routing and effective depth as context length changes.
- Prefill latency, Decode latency, memory use, and throughput.

Use controlled synthetic tasks alongside established benchmarks such as NoLiMa or RULER. Length is an experimental variable for identifying scaling behavior, not the primary claimed contribution.

### Stage D: Architecture Generalization

After the core mechanism is validated, extend experiments to larger open models and alternative Attention families.

FlashAttention should be treated as an execution optimization for exact Attention, whereas Linear, Sparse, and Hybrid Attention represent different computational mechanisms. These dimensions should be evaluated separately before being combined.

## 7. Theoretical Analysis

The initial theory effort will focus on properties that can be stated precisely.

**Function class nesting.** If a larger expert bank preserves the original experts and routing can continue selecting the original configuration, the larger architecture can represent the original model's function. This establishes a non decreasing representational capacity under the nesting assumption. It does not prove that optimization will improve or maintain generalization.

**Active compute.** Under the shared KV design, fixed \(K\) bounds the number of executed expert paths per token independently of \(E\). Total parameter growth can therefore be separated from expert activation count, although router and execution costs must still be measured.

**Depth and expressiveness.** Analyze the interaction between parameterized paths, serial depth, and sequence function composition. Prior work, including *A Provable Expressiveness Hierarchy in Hybrid Linear-Full Attention* (ICML 2026), establishes theoretical distinctions between Full, Linear, and Hybrid Attention in specific compositional tasks. The relationship between conditional activation and depth remains an empirical and theoretical question for this project.

The initial aim is a clear capacity and cost model, rather than claiming a universal expressivity theorem.

## 8. Evaluation and Success Criteria

The principal success criterion is a reproducible quality improvement at matched active computation as total Attention capacity increases.

Supporting evidence should include:

- Distinct expert functions established through utilization analysis and ablations.
- Gains beyond those explained by additional training compute or total parameters alone.
- Stable routing behavior across seeds and relevant context lengths.
- Measurable improvements in actual execution efficiency where conditional execution is supported.
- A clear account of where gains saturate or disappear.

Failure to obtain these results would still help establish whether parameterized Attention experts provide useful capacity scaling under the studied conditions.

## 9. Scope and Longer Term Direction

The first study focuses on conditional Attention capacity. Adaptive depth is the second stage, followed by context length extrapolation.

Persistent history access and learned memory management remain a separate longer term direction. In that setting, the working context may stay bounded while the accessible history grows through memory, retrieval, and state updates. This is related to the broader objective of efficient information processing, but should not be combined with the initial architecture study before its core hypothesis is validated.

## References

1. Zhang et al. *Mixture of Attention Heads: Selecting Attention Heads Per Token*. EMNLP 2022. https://aclanthology.org/2022.emnlp-main.278/
2. Yuan et al. *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*. 2025. https://arxiv.org/abs/2502.11089
3. Dao et al. *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS 2022. https://arxiv.org/abs/2205.14135
4. Ye et al. *A Provable Expressiveness Hierarchy in Hybrid Linear-Full Attention*. ICML 2026. https://proceedings.mlr.press/v306/ye26h.html
5. Mohammed. *Adaptive Computation Depth via Learned Token Routing in Transformers*. 2026 preprint. https://arxiv.org/abs/2605.05222

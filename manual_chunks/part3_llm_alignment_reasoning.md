# Part 3: LLM Alignment, Post-Training & Reasoning (Chapters 15–21)
## Exhaustive Technical Extraction & Reference Manual

> **Scope**: Comprehensive extraction across 34 markdown chapters in `hands-on-modern-rl`:
> - Chapter 15: RLHF (Reinforcement Learning from Human Feedback) & veRL PPO
> - Chapter 16: Industrial LLM Post-Training Pipelines & Distributed Systems
> - Chapter 17: Direct Preference Optimization (DPO) & Offline Preference Family
> - Chapter 18: Group Relative Policy Optimization (GRPO), DAPO, RLVR, OPD & Sandbox Environments
> - Chapter 19: Reasoning Models, R1-Zero Emergence, Test-Time Compute Scaling & Hybrid Thinking
> - Chapter 20: Process Reward Models (PRM), Formal Verifiers & Tree Search
> - Chapter 21: Constitutional AI (CAI), RLAIF & HHH Alignment
>
> **Target Output**: `/home/codenuclei/hands-on-modern-rl/manual_chunks/part3_llm_alignment_reasoning.md`

---

# Table of Contents
1. [Core Mathematical Formulations & Derivations](#1-core-mathematical-formulations--derivations)
   - 1.1 SFT & Imitation Learning Foundations
   - 1.2 Bradley-Terry Preference Modeling & Reward Model Loss
   - 1.3 PPO-RLHF: Token-Level vs Sequence-Level Rewards, KL Penalty & GAE
   - 1.4 Direct Preference Optimization (DPO): Complete 3-Step Derivation & Implicit Reward
   - 1.5 The Offline Preference Optimization Family: KTO, IPO, SimPO, SLiC, ORPO
   - 1.6 Group Relative Policy Optimization (GRPO): Core Formulation & Mechanisms
   - 1.7 Advanced GRPO Variants (2025–2026): Dr.GRPO, DeepSeek V3.2, GSPO, CISPO, VAPO, DAPO, RPT
   - 1.8 On-Policy Distillation (OPD): Reverse KL, Token-Level Rewards & Exposure Bias
   - 1.9 Process Reward Models (PRM), Step-Level Objectives & Formal Verifiers
   - 1.10 Search During Reasoning, Test-Time Compute Scaling & MCTS
   - 1.11 Constitutional AI (CAI), RLAIF & Formalized HHH Alignment
2. [Industrial Training Systems, Distributed Frameworks & Configurations](#2-industrial-training-systems-distributed-frameworks--configurations)
   - 2.1 veRL Distributed Execution Architecture & HybridFlow
   - 2.2 veRL PPO GSM8K Complete Production Recipe & Hyperparameters
   - 2.3 veRL Code Sandbox RLVR Production Recipe
   - 2.4 Financial Tool-Calling GRPO + TRL Recipe
   - 2.5 Industrial Post-Training Architectures Across Frontier Labs
   - 2.6 Compute Estimation, FLOPs Modeling & Asynchronous Training Systems
3. [Pathology Analysis: Failure Modes & Mitigation Mechanics](#3-pathology-analysis-failure-modes--mitigation-mechanics)
   - 3.1 Length Bias & Verbosity Explosion
   - 3.2 Reward Hacking & Goodhart's Law
   - 3.3 Variance Collapse & The Graduation Problem
   - 3.4 KL Drift, Distribution Shift & Mode Collapse
   - 3.5 Sycophancy, Over-Refusal & Format Degeneracy
4. [Empirical Benchmarks, Case Studies & Comparative Matrices](#4-empirical-benchmarks-case-studies--comparative-matrices)
   - 4.1 EvalScope Benchmark Results (GSM8K, HumanEval, LiveCodeBench)
   - 4.2 Reasoning & Math Benchmarks (AIME 2024, MATH, PRM800K)
   - 4.3 DPO vs PPO vs GRPO Architectural & Resource Comparison Matrix
   - 4.4 Complete Algorithm Taxonomy & Decision Tree

---

# 1. Core Mathematical Formulations & Derivations

### 1.1 SFT & Imitation Learning Foundations

In modern post-training, language models transition from raw next-token predictors (continuation engines) to instruction-following assistants. The pretraining objective maximizes log-likelihood across unconstrained text corpora:

$$\mathcal{L}_{\text{LM}}(\theta) = -\sum_{t=1}^{T} \log \pi_\theta(x_t \mid x_{<t})$$

Generation is framed as a Markov Decision Process (MDP) $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$:
- **State $s_t$**: The prompt $x$ concatenated with all previously generated tokens $y_{<t}$: $s_t = (x, y_1, \ldots, y_{t-1})$.
- **Action $a_t$**: The choice of the next token $y_t \in \mathcal{V}$ from the vocabulary $\mathcal{V}$ (typically $|\mathcal{V}| \approx 32{,}000$ to $150{,}000$).
- **Policy $\pi_\theta(a_t \mid s_t)$**: $P_\theta(y_t \mid x, y_{<t})$.
- **Transition $P(s_{t+1} \mid s_t, a_t)$**: Deterministic sequence concatenation: $s_{t+1} = (s_t, a_t)$.
- **Reward $R(s_t, a_t)$**: Evaluated from human preferences or programmatic verification rules.

Supervised Fine-Tuning (SFT) is Behavior Cloning (BC) on human or synthetic demonstrations $\mathcal{D}_{\text{SFT}} = \{(x, y)\}$:

$$\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{SFT}}} \left[\sum_{t=1}^{|y|} m_t \log \pi_\theta(y_t \mid x, y_{<t})\right]$$

where $m_t \in \{0, 1\}$ is a binary mask set to $1$ only on assistant response tokens, preventing the model from computing loss or updating parameters on prompt tokens.

#### Fatal Flaw of Behavior Cloning: Exposure Bias & Distribution Shift
BC assumes training trajectories are drawn from the expert distribution $p_{\text{expert}}(s)$. During autoregressive inference, an error at step $t$ shifts the context $s_{t+1}$ outside the expert support. Errors compound quadratically along sequence length $T$ ($O(\epsilon T^2)$), causing rapid hallucination or repetition loops.

---

### 1.2 Bradley-Terry Preference Modeling & Reward Model Loss

When subjective human feedback cannot be captured by simple heuristics, preference data $\mathcal{D}_{\text{pref}} = \{(x, y_w, y_l)\}$ is collected, where $y_w$ is the winning (chosen) response and $y_l$ is the losing (rejected) response given prompt $x$.

Under the **Bradley-Terry (BT) preference model**, the probability that response $y_w$ is preferred over $y_l$ is governed by an underlying scalar latent reward $r^*(x, y)$:

$$P(y_w \succ y_l \mid x) = \sigma\left(r^*(x, y_w) - r^*(x, y_l)\right) = \frac{1}{1 + \exp\left(-(r^*(x, y_w) - r^*(x, y_l))\right)} = \frac{\exp(r^*(x, y_w))}{\exp(r^*(x, y_w)) + \exp(r^*(x, y_l))}$$

A Reward Model $r_\phi(x, y)$ parameterized by $\phi$ is trained via negative log-likelihood:

$$\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{pref}}} \left[\log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right]$$

For preference pairs annotated with a margin $m(x, y_w, y_l) \ge 0$, the margin-augmented loss is:

$$\mathcal{L}_{\text{RM-margin}}(\phi) = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l) - m(x, y_w, y_l)\right)\right]$$

#### Multi-Component Composite Reward Formulation
In production RLHF, the raw RM output is combined with explicit rule-based rewards:

$$R_{\text{mix}}(x, y) = R_{\text{RM}}(x, y) + \alpha R_{\text{format}}(x, y) + \beta R_{\text{length}}(x, y) + \gamma R_{\text{correctness}}(x, y)$$

---

### 1.3 PPO-RLHF: Token-Level vs Sequence-Level Rewards, KL Penalty & GAE

In classic PPO-RLHF, the policy $\pi_\theta$ is optimized to maximize reward while remaining anchored to a reference model $\pi_{\text{ref}}$ (the frozen SFT checkpoint) to prevent reward hacking and catastrophic forgetting:

$$\max_\theta \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(\cdot \mid x)} \left[ r_\phi(x, y) - \beta D_{\text{KL}}\left(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\right) \right]$$

#### Per-Token Reward Formulation
To provide step-level credit assignment, the sequence reward is converted into per-token rewards $r_t$:

$$r_t = \begin{cases} -\beta \left(\log \pi_\theta(y_t \mid s_t) - \log \pi_{\text{ref}}(y_t \mid s_t)\right), & t < T \ r_{\text{RM}}(x, y) - \beta \left(\log \pi_\theta(y_T \mid s_T) - \log \pi_{\text{ref}}(y_T \mid s_T)\right), & t = T \end{cases}$$

Here, $\beta$ is the KL penalty coefficient. If the Actor places higher probability on token $y_t$ than the Reference model ($\log \pi_\theta > \log \pi_{\text{ref}}$), it incurs a negative penalty step.

#### Generalized Advantage Estimation (GAE)
A Critic network $V_\psi(s_t)$ predicts the expected return from intermediate token prefix $s_t$. The temporal difference (TD) residual at step $t$ is:

$$\delta_t = r_t + \gamma V_\psi(s_{t+1}) - V_\psi(s_t)$$

In language generation, $\gamma = 1.0$ is standard. The GAE advantage $\hat{A}_t^{\text{GAE}(\gamma, \lambda)}$ is computed recursively:

$$\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{T - t - 1} (\gamma \lambda)^l \delta_{t+l} = \delta_t + (\gamma \lambda) \hat{A}_{t+1}^{\text{GAE}}$$

#### Clipped Surrogate Policy Gradient & Value Loss
Let $\rho_t(\theta)$ denote the importance sampling ratio between the updated and rollout policies:

$$\rho_t(\theta) = \frac{\pi_\theta(y_t \mid s_t)}{\pi_{\theta_{\text{old}}}(y_t \mid s_t)} = \exp\left(\log \pi_\theta(y_t \mid s_t) - \log \pi_{\theta_{\text{old}}}(y_t \mid s_t)\right)$$

The PPO clipped objective is:

$$\mathcal{L}_{\text{clip}}(\theta) = -\mathbb{E}_t \left[ \min\left(\rho_t(\theta) \hat{A}_t, \, \operatorname{clip}(\rho_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t\right) \right]$$

The Critic is trained via Mean Squared Error against empirical returns $R_t = \hat{A}_t + V_{\psi_{\text{old}}}(s_t)$, optionally clipped:

$$\mathcal{L}_{\text{value}}(\psi) = \frac{1}{2} \mathbb{E}_t \left[ \max\left( (V_\psi(s_t) - R_t)^2, \, (V_{\psi_{\text{old}}}(s_t) + \operatorname{clip}(V_\psi(s_t) - V_{\psi_{\text{old}}}(s_t), -\epsilon_v, \epsilon_v) - R_t)^2 \right) \right]$$

---

### 1.4 Direct Preference Optimization (DPO): Complete 3-Step Derivation & Implicit Reward

Direct Preference Optimization (Rafailov et al., 2023) eliminates the need to train a separate Reward Model and Critic, solving the RLHF optimization problem analytically.

#### The Primal Constrained RL Problem
Given prompt $x \sim \mathcal{D}$, the KL-constrained RL objective is:

$$\max_\pi \mathbb{E}_{x \sim \mathcal{D}} \left[ \mathbb{E}_{y \sim \pi(\cdot \mid x)} [r(x, y)] - \beta D_{\text{KL}}\left(\pi(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\right) \right]$$

Expanding the expectation and KL definition:

$$\max_\pi \sum_{y} \pi(y \mid x) r(x, y) - \beta \sum_{y} \pi(y \mid x) \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$

$$= \max_\pi -\beta \sum_y \pi(y \mid x) \left[ \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} - \frac{1}{\beta} r(x, y) \right]$$

$$= \min_\pi \sum_y \pi(y \mid x) \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)}$$

#### Step 1: Analytical Closed-Form Optimal Policy
Define the unnormalized distribution $\tilde{\pi}^*(y \mid x) = \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$ and partition function:

$$Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

Multiplying and dividing by $Z(x)$:

$$\sum_y \pi(y \mid x) \log \frac{\pi(y \mid x)}{\frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)} - \log Z(x) = D_{\text{KL}}\left(\pi(y \mid x) \,\|\, \pi^*(y \mid x)\right) - \log Z(x)$$

where the optimal policy $\pi^*$ is given by Gibbs/Boltzmann distribution:

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

Since $D_{\text{KL}} \ge 0$ with equality if and only if distributions match, $\pi^*(y \mid x)$ uniquely minimizes the objective.

#### Step 2: Inverting Policy to Express Ground-Truth Reward
Taking the natural logarithm of both sides of the optimal policy equation:

$$\log \pi^*(y \mid x) = \log \pi_{\text{ref}}(y \mid x) + \frac{1}{\beta} r(x, y) - \log Z(x)$$

Rearranging for the reward $r(x, y)$:

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

This establishes that **the reward function is uniquely determined by the log-ratio of the optimal policy to the reference policy, up to a prompt-dependent partition function $Z(x)$**.

#### Step 3: Substitution into the Bradley-Terry Objective
Recall the BT preference likelihood:

$$P(y_w \succ y_l \mid x) = \sigma\left(r(x, y_w) - r(x, y_l)\right)$$

Substitute the reward expression into the difference:

$$r(x, y_w) - r(x, y_l) = \left[ \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \beta \log Z(x) \right] - \left[ \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} + \beta \log Z(x) \right]$$

Crucially, **the partition function $Z(x)$ cancels out completely**:

$$r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}$$

Replacing $\pi^*$ with parameterized policy $\pi_\theta$, the **DPO Loss Objective** is:

$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

#### DPO Implicit Reward & Gradient Dynamics
The implicit reward defined by DPO is:

$$\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$

Taking the gradient of $\mathcal{L}_{\text{DPO}}(\theta)$ with respect to $\theta$:

$$
abla_\theta \mathcal{L}_{\text{DPO}}(\theta) = -\beta \mathbb{E}_{(x, y_w, y_l)} \left[ \sigma\left(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)\right) \left[ 
abla_\theta \log \pi_\theta(y_w \mid x) - 
abla_\theta \log \pi_\theta(y_l \mid x) \right] \right]$$

The weight $\sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w)) = 1 - \sigma(\hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l))$ acts as an adaptive learning rate: when the model incorrectly assigns higher implicit reward to the rejected response $y_l$, the weight approaches $1$, providing maximal gradient to correct the error. When $y_w$ is already strongly preferred, the weight decays to $0$.

---

### 1.5 The Offline Preference Optimization Family: KTO, IPO, SimPO, SLiC, ORPO

Following DPO, multiple offline formulations eliminated different architectural bottlenecks:

#### 1. KTO (Kahneman-Tversky Optimization)
Designed for unpaired preference data where samples have binary upvote/downvote signals $(y, \text{label}) \in \{(y, +1), (y, -1)\}$ based on Prospect Theory's loss aversion:

$$\mathcal{L}_{\text{KTO}}(\theta) = \mathbb{E}_{x, y} \left[ w(y) \left( 1 - v_{\text{KTO}}\left( \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} - z_{\text{ref}} \right) \right) \right]$$

where $v_{\text{KTO}}(x) = \sigma(x)$ for positive samples and $\sigma(-\lambda_{\text{loss}} x)$ for negative samples ($\lambda_{\text{loss}} > 1$, reflecting human cognitive asymmetry where losses hurt more than equal gains), and $z_{\text{ref}}$ is a baseline offset.

#### 2. IPO (Identity Preference Optimization - Azar et al., 2024)
Addresses DPO's vulnerability to deterministic preference overfitting when log-ratios grow unbounded. IPO optimizes the unregularized BT loss with an explicit $L_2$ penalty:

$$\mathcal{L}_{\text{IPO}}(\theta) = \mathbb{E}_{(x, y_w, y_l)} \left[ \left( \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} - \frac{1}{2\tau} \right)^2 \right]$$

#### 3. SimPO (Simple Preference Optimization - Meng et al., 2024)
Removes the Reference Model $\pi_{\text{ref}}$ entirely to save 50% memory, and incorporates a length-normalized log-probability with a target margin $\gamma > 0$:

$$\mathcal{L}_{\text{SimPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \frac{\beta}{|y_w|} \log \pi_\theta(y_w \mid x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l \mid x) - \gamma \right) \right]$$

#### 4. SLiC (Sequence Likelihood Calibration)
Uses a hinge margin loss combined with an SFT regularization term:

$$\mathcal{L}_{\text{SLiC}}(\theta) = \mathbb{E}_{(x, y_w, y_l)} \left[ \max\left(0, \, \delta - \log \pi_\theta(y_w \mid x) + \log \pi_\theta(y_l \mid x)\right) \right] - \lambda \mathbb{E}_{(x, y_w)} [\log \pi_\theta(y_w \mid x)]$$

#### 5. ORPO (Odds Ratio Preference Optimization)
Unifies SFT and preference alignment into a single loss without a reference model, using the odds ratio $\text{odds}_\theta(y \mid x) = \frac{P_\theta(y \mid x)}{1 - P_\theta(y \mid x)}$:

$$\mathcal{L}_{\text{ORPO}}(\theta) = \mathcal{L}_{\text{SFT}}(\theta) + \lambda_{\text{OR}} \mathcal{L}_{\text{OR}}(\theta) = -\log \pi_\theta(y_w \mid x) - \lambda_{\text{OR}} \log \sigma\left( \log \frac{\text{odds}_\theta(y_w \mid x)}{\text{odds}_\theta(y_l \mid x)} \right)$$

---

### 1.6 Group Relative Policy Optimization (GRPO): Core Formulation & Mechanisms

GRPO (Shao et al., 2024; DeepSeekMath / DeepSeek-R1) eliminates the Critic network from PPO. Instead of training a separate value network $V_\psi(s)$ to estimate baseline returns, GRPO samples a group of $G$ responses $\{y_1, y_2, \ldots, y_G\}$ for each prompt $q$ from the old policy $\pi_{\theta_{\text{old}}}$, evaluates each with a reward function $R(q, y_i) = r_i$, and standardizes rewards within the group.

#### Within-Group Advantage Normalization
The advantage $\hat{A}_i$ for the $i$-th response is computed directly via empirical group statistics:

$$\bar{r} = \frac{1}{G} \sum_{j=1}^{G} r_j, \qquad \sigma_r = \sqrt{\frac{1}{G} \sum_{j=1}^{G} (r_j - \bar{r})^2 + \epsilon}$$

$$\hat{A}_i = \frac{r_i - \bar{r}}{\sigma_r}$$

When all responses in a group receive identical rewards (e.g. all correct $r=1$ or all failed $r=0$), $\sigma_r \approx 0$ and $\hat{A}_i$ is clamped to $0$, avoiding numerical explosion.

#### GRPO Clipped Objective with Reference KL Regularization
Unlike sequence-level methods that compress the ratio, standard GRPO preserves the 2D token tensor $[B \times G, T]$. The importance ratio at token $t$ is:

$$\rho_{i,t}(\theta) = \frac{\pi_\theta(y_{i,t} \mid q, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid q, y_{i,<t})}$$

The objective maximizes advantage while penalizing per-token KL divergence to a reference policy $\pi_{\text{ref}}$:

$$\mathcal{J}_{\text{GRPO}}(\theta) = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \left[ \min\left( \rho_{i,t}(\theta) \hat{A}_i, \, \operatorname{clip}(\rho_{i,t}(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_i \right) - \beta D_{\text{KL}}\left(\pi_\theta(\cdot \mid s_{i,t}) \,\|\, \pi_{\text{ref}}(\cdot \mid s_{i,t})\right) \right]$$

where the per-token KL divergence is approximated using Schulman's unbiased estimator:

$$D_{\text{KL}}\left(\pi_\theta \,\|\, \pi_{\text{ref}}\right) \approx \frac{\pi_{\text{ref}}(y_{i,t} \mid s_{i,t})}{\pi_\theta(y_{i,t} \mid s_{i,t})} - \log \frac{\pi_{\text{ref}}(y_{i,t} \mid s_{i,t})}{\pi_\theta(y_{i,t} \mid s_{i,t})} - 1$$

---

### 1.7 Advanced GRPO Variants (2025–2026): Dr.GRPO, DeepSeek V3.2, GSPO, CISPO, VAPO, DAPO, RPT

#### 1. Dr.GRPO (Liu et al., 2025 - arXiv:2503.20783)
Discovered that dividing by the group standard deviation $\sigma_r$ induces severe **Length Bias** and incentivizes **Reward Hacking**. When variance is small, dividing by $\sigma_r \to 0$ artificially amplifies advantages; the model learns to increase variance by generating longer, verbose responses. Dr.GRPO removes standard deviation normalization entirely:

$$\tilde{r}_i^{\text{Dr.GRPO}} = r_i - \frac{1}{G}\sum_{j=1}^G r_j$$

#### 2. DeepSeek V3.2 (arXiv:2512.02556)
Extends Dr.GRPO with domain-specific rules:
- **Zero KL for Math Tasks**: Setting $\beta = 0$ for verifiable mathematics and coding, allowing the model to explore reasoning structures without reference constraints.
- Format strictness checks and dynamic difficulty scheduling.

#### 3. GSPO (Group Sequence Policy Optimization - Qwen3 Team, arXiv:2507.18071)
Used across the entire Qwen3 series (Qwen3-235B-A22B MoE, Qwen3-Thinking, Qwen3-Coder). Standard token-level importance sampling has massive variance across 10,000-card MoE clusters. GSPO introduces the **geometric mean sequence-level ratio**:

$$s_i(\theta) = \left( \frac{\pi_\theta(y_i \mid x)}{\pi_{\theta_{\text{old}}}(y_i \mid x)} \right)^{1/|y_i|} = \exp\left( \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \log \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid x, y_{i,<t})} \right) = \left( \prod_{t=1}^{|y_i|} \rho_{i,t} \right)^{1/|y_i|}$$

The sequence-level clipped objective is:

$$\mathcal{J}^{\text{GSPO}}(\theta) = \mathbb{E}\left[ \min\left( s_i(\theta) \hat{A}_i, \, \operatorname{clip}(s_i(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_i \right) \right]$$

The $1/|y_i|$ geometric mean normalization is mathematically essential: without it, the raw product $\prod_t \rho_t$ collapses or explodes exponentially with response length.

#### 4. CISPO (MiniMax M1 - arXiv:2506.13585)
Traditional PPO/GRPO clipping zeros out gradients when ratios exceed bounds, causing tokens to stall. MiniMax's CISPO modifies the pruning object into an **importance sampling weight** applied to the advantage, reversing the ratio:

$$\tilde{\rho}_t = \operatorname{clip}\left( \frac{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}{\pi_\theta(a_t \mid s_t)}, \, 1 - \epsilon, \, 1 + \epsilon \right)$$

This reversed ratio scales the gradient contribution rather than truncating it, preserving gradient flow. CISPO solves precision drift in low-precision training under MiniMax's Lightning Attention architecture on 512 H800 GPUs.

#### 5. VAPO (Value-based Augmented PPO - ByteDance Seed, arXiv:2504.05118)
Demonstrates that in ultra-long Chain-of-Thought (CoT) reasoning, Critic-free GRPO suffers from coarse sequence-level credit assignment (the whole 4,000-token trace gets one scalar advantage). VAPO reintroduces a lightweight Value Model to provide token-level advantage estimation $\hat{A}_t$, proving that Critic-free RL was an engineering concession rather than a theoretical optimum.

#### 6. DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization - NeurIPS 2025)
Introduces four orthogonal improvements to GRPO:
- **Clip-Higher**: Asymmetric clipping ranges (e.g. $[1-\epsilon_{\text{low}}, 1+\epsilon_{\text{high}}]$ where $\epsilon_{\text{high}} > \epsilon_{\text{low}}$) to avoid suppressing promising low-probability exploratory tokens.
- **Dynamic Sampling**: Actively filters out prompts where the policy has achieved accuracy $\ge 0.95$ (zero gradient variance), cutting training steps by 50% on AIME 2024 to reach score 50.
- **Token-Level Loss**: Fine-grained credit assignment distinguishing valid reasoning steps from late arithmetic slips.
- **Overlong Reward Shaping**: Replaces hard length truncation penalties with smooth differentiable decay:
  $$R_{\text{len}}(y) = -\alpha \max(0, |y| - L_{\text{target}})^p$$

#### 7. RPT (Reinforcement Pre-training - Microsoft Research, arXiv:2506.08007)
Integrates RL into the pre-training stage by conditioning next-token prediction on generated internal reasoning tokens:

$$\mathcal{L}_{\text{RPT}}(\theta) = -\mathbb{E}\left[ \log \pi_\theta(a_t \mid s_{<t}, \text{reasoning}_t) \right] + \beta \mathcal{L}_{\text{RL}}$$

---

### 1.8 On-Policy Distillation (OPD): Reverse KL, Token-Level Rewards & Exposure Bias

On-Policy Distillation (GKD - Agarwal et al., 2024; MiniLLM; DistiLLM) resolves the distribution shift inherent in standard Sequence-level Knowledge Distillation (SeqKD). In SeqKD (off-policy), the student trains on teacher-generated contexts:

$$\mathcal{L}_{\text{hard KD}}(\theta) = -\sum_{t} \log \pi_\theta(y_t^T \mid x, y_{<t}^T)$$

$$\mathcal{L}_{\text{soft KD}}(\theta) = \sum_t D_{\text{KL}}\left( q(\cdot \mid c_t^T) \,\|\, \pi_\theta(\cdot \mid c_t^T) \right)$$

In OPD, **trajectories are sampled directly from the student policy** $y \sim \pi_\theta(\cdot \mid x)$, and the teacher $q$ evaluates each student token online:

$$\mathcal{L}_{\text{OPD}}(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \left[ \sum_{t=1}^{|y|} \log \frac{\pi_\theta(y_t \mid c_t)}{q(y_t \mid c_t)} \right]$$

This minimizes reverse KL on the student's visited state space:

$$\min_\theta \mathbb{E}_{s \sim d^{\pi_\theta}} \left[ D_{\text{KL}}\left(\pi_\theta(\cdot \mid s) \,\|\, q(\cdot \mid s)\right) \right]$$

#### Forward KL vs Reverse KL Dynamics
- **Forward KL $D_{\text{KL}}(q \,\|\, \pi_\theta)$**: Zero-avoiding / mean-seeking. Forces $\pi_\theta > 0$ wherever $q > 0$. In multi-modal generative tasks, this produces blurry distributions and hallucinatory off-distribution outputs.
- **Reverse KL $D_{\text{KL}}(\pi_\theta \,\|\, q)$**: Mode-seeking. Forces $\pi_\theta \to 0$ wherever $q \to 0$. The student concentrates probability mass exclusively on modes approved by the teacher.

#### OPD Token-Level Reward in RL Formulation
OPD can be executed directly as an RL algorithm where the teacher log-probability provides a dense token-level reward:

$$r_t^{\text{OPD}} = \log q(y_t \mid x, y_{<t}) - \log \pi_\theta(y_t \mid x, y_{<t})$$

The advantage estimate is $\hat{A}_t = r_t^{\text{OPD}}$. If the teacher approves the student's chosen token more than the student did, $\hat{A}_t > 0$ and the token is reinforced.

---

### 1.9 Process Reward Models (PRM), Step-Level Objectives & Formal Verifiers

#### Outcome RM (ORM) vs Process RM (PRM)
- **Outcome RM**: Evaluates complete response: $r_{\text{ORM}} = \text{ORM}(q, o) \in \mathbb{R}$.
- **Process RM**: Evaluates the validity of each reasoning step $i$: $r_i = \text{PRM}(q, o_{\le i}, s_i) \in \mathbb{R}$.

#### Discriminative PRM Loss (PRM800K - Lightman et al., 2023)
Reasoning is split into steps using delimiter tags (`\n\n` or `\boxed{}`). Each step $s_i$ is classified as $c \in \{\text{good}, \text{bad}, \text{neutral}\}$:

$$\mathcal{L}_{\text{PRM}}(\phi) = -\sum_{i=1}^{K} \sum_{c \in \{\text{good}, \text{bad}, \text{neutral}\}} y_{i,c} \log P_\phi(c \mid q, o_{\le i}, s_i)$$

#### Step-Level Advantage Aggregation
Step scores are aggregated into response rewards via product or minimum heuristics:

$$r_{\text{process}}^{\text{prod}} = \prod_{i=1}^K P_\phi(\text{good} \mid s_i), \qquad r_{\text{process}}^{\text{min}} = \min_{i=1..K} P_\phi(\text{good} \mid s_i)$$

Composite step-outcome reward:

$$r_{\text{total}} = \alpha r_{\text{process}} + \beta r_{\text{outcome}}$$

#### Generative PRM & ThinkPRM
Generates natural language critique tokens followed by a score token:

$$\text{GenRM}(q, o) = P_\phi(\text{"good"} \mid q, o, \text{"Critique: ... Evaluation:"})$$

#### Formal Verifiers (AlphaProof, AlphaGeometry, DeepSeek-Prover-V2)
Interfacing with formal interactive theorem provers (Lean 4, Isabelle, Coq):
- **Deterministic soundness**: A proof checked by Lean 4 kernel has formal mathematical correctness ($r \in \{0, 1\}$), eliminating verifier hallucinations.
- **Autoformalization**: Translating informal natural language problems into formal Lean 4 statements before proof tree search.

---

### 1.10 Search During Reasoning, Test-Time Compute Scaling & MCTS

#### Test-Time Compute Scaling Law (Snell et al., 2024)
For independent parallel rollouts (Best-of-$N$) with single-sample success probability $p$:

$$P(\text{At least one correct}) = 1 - (1 - p)^N$$

Test-time compute can be spent along two orthogonal axes:
1. **Serial Depth**: Long internal Chain-of-Thought with self-reflection and backtracking.
2. **Parallel Breadth**: Generating $N$ candidates aggregated via Majority Voting, Best-of-$N$ (ORM/PRM reranking), or PaCoRe.

#### Monte Carlo Tree Search (MCTS) UCB / PUCT Formula
In tree-based reasoning search (AlphaZero-style LLM search), nodes represent step prefixes $n = (q, s_{\le i})$. The Upper Confidence Bound applied to Trees (UCB1) is:

$$\text{UCB}(n) = Q(n) + c \cdot \sqrt{\frac{\ln N(p)}{N(n)}}$$

Under the Predictor Upper Confidence Tree (PUCT) formulation:

$$\text{PUCT}(s, a) = Q(s, a) + c_{\text{puct}} \cdot P(s, a) \cdot \frac{\sqrt{\sum_{b} N(s, b)}}{1 + N(s, a)}$$

where:
- $Q(s, a)$: Average process reward from rollouts starting from node $(s, a)$.
- $P(s, a)$: Prior policy probability $\pi_\theta(a \mid s)$ from the language model.
- $N(s, a)$: Visit count of edge $(s, a)$.
- $N(p)$: Total visit count of parent node $p$.

#### PaCoRe (Parallel CoT Reasoning) & Generative Aggregation
Multiple heterogeneous reasoning chains (algebraic, geometric, computational) are generated concurrently. A coordinator LLM inspects all traces and synthesizes a verified consensus:

$$\hat{y}^* = \arg\max_y P_{\text{coord}}(y \mid x, \tau_1, \tau_2, \ldots, \tau_K)$$

---

### 1.11 Constitutional AI (CAI), RLAIF & Formalized HHH Alignment

Constitutional AI (Anthropic - Bai et al., 2022) replaces human feedback with principle-driven AI supervision.

#### 1. Supervised Stage (SL-CAI: Critique & Revision)
Given harmful or unhelpful response $y_0$ to prompt $x$, and constitution principle $c_k$:

$$\text{critique} \sim \pi_\theta(\cdot \mid x, y_0, c_k, \text{"critique:"})$$

$$y^* \sim \pi_\theta(\cdot \mid x, y_0, \text{critique}, c_k, \text{"revision:"})$$

The model is fine-tuned on revised responses:

$$\mathcal{L}_{\text{SL-CAI}}(\theta) = -\mathbb{E}_{(x, y_0, c_k)} \left[ \log \pi_\theta(y^* \mid x) \right]$$

#### 2. Reinforcement Learning Stage (RL-CAI / RLAIF)
An AI judge evaluates candidate pairs $(y_1, y_2)$ using constitution prompt $J(x, y_1, y_2, c_k)$:

$$P_{\text{AI}}(y_1 \succ y_2) = \frac{\exp(r_\phi(x, y_1))}{\exp(r_\phi(x, y_1)) + \exp(r_\phi(x, y_2))}$$

Preference RM is trained on synthetic dataset $\mathcal{D}_{\text{AI}}$:

$$\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}_{\text{AI}}} \left[ \log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right) \right]$$

Followed by standard PPO/DPO policy optimization:

$$R_{\text{RLAIF}}(x, y) = r_\phi(x, y) - \beta D_{\text{KL}}\left(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\right)$$

#### 3. Formalized HHH Alignment Objectives
The three core alignment pillars are formalized mathematically:
- **Helpful**: Maximizes expected user utility function $U_u(x, y)$:
  $$\text{Helpful}(y \mid x) = \mathbb{E}_{u \sim \text{user}} \left[ U_u(x, y) \right]$$
  Adjusted for length bias: $r_{\text{adj}}(x, y) = r_\phi(x, y) - \lambda_{\text{len}} \cdot |y|$.
- **Harmless**: Minimizes probability of real-world harm:
  $$\text{Harmless}(y \mid x) = 1 - \mathbb{P}(\text{harm} \mid x, y)$$
- **Honest**: Minimizes epistemic divergence from objective ground truth:
  $$\text{Honest}(y \mid x) = 1 - D_{\text{KL}}\left( p_{\text{model}}(\cdot \mid x) \,\|\, p_{\text{true}}(\cdot \mid x) \right)$$

#### Joint HHH Multi-Objective Optimization
$$r_{\text{HHH}}(x, y) = \alpha_H \cdot \text{Helpful}(y \mid x) + \alpha_{\text{HL}} \cdot \text{Harmless}(y \mid x) + \alpha_{\text{HN}} \cdot \text{Honest}(y \mid x)$$

In Claude 3.5 production systems, this is unified with RLVR:

$$R_{\text{Claude 3.5}}(x, y) = w_{\text{task}} r_{\text{RLVR}}(x, y) + w_{\text{safe}} r_{\text{CAI}}(x, y) + w_{\text{honest}} r_{\text{calibration}}(x, y)$$

Auditing and context-consistency losses:

$$\mathcal{L} = \mathcal{L}_{\text{pref}} + \lambda_{\text{ctx}} \mathcal{L}_{\text{context-consistency}}$$

$$\mathcal{L}_{\text{audit}} = -\mathbb{E}\left[ \log \sigma\left(r_\phi(x, y_w, a_w) - r_\phi(x, y_l, a_l)\right) \right]$$

where $a(y) \in \mathcal{P}(\text{Constitution})$ provides attribution mapping to constitutional articles.

---

# 2. Industrial Training Systems, Distributed Frameworks & Configurations

### 2.1 veRL Distributed Execution Architecture & HybridFlow

veRL (VolcEngine Reinforcement Learning - arXiv:2409.19256) decouples RLHF into a **HybridFlow** execution model:
- **Disaggregation of Rollout vs Training**:
  - Rollout (Actor inference) is dominated by memory bandwidth, KV caching, and continuous batching, optimal on vLLM engines with Tensor Parallelism (TP).
  - Model Training (Actor update, Critic update) is compute-bound, optimal on Megatron-LM or PyTorch FSDP with Pipeline Parallelism (PP) and ZeRO-3 parameter sharding.
- **ResourcePool**: veRL assigns models dynamically to physical GPU pools using Ray actors:
  - `actor_rollout_ref`: Co-locates Actor, Rollout engine, and Reference model on shared GPUs. When transitioning from generation to training, vLLM frees its KV cache buffer, letting FSDP claim memory for forward/backward passes.
  - `critic`: Co-locates Critic and Reward Model on dedicated or shared workers.

---

### 2.2 veRL PPO GSM8K Complete Production Recipe & Hyperparameters

From `code/chapter08_rlhf/verl_gsm8k/`:
- Launcher: `run_qwen2_5_0_5b_ppo_single_gpu.sh` & `run_qwen2_5_0_5b_ppo_8gpu.sh`
- Verifier: `gsm8k_reward.py`

#### Complete Parameter Configuration Table
| Component | Parameter | Single-GPU Value (0.5B) | 8-GPU Production Value (7B) | Explanation |
| :--- | :--- | :--- | :--- | :--- |
| **Data** | `data.train_files` | `$DATA_DIR/train.parquet` | `$DATA_DIR/train.parquet` | GSM8K parquet with prompt & ground_truth |
| | `data.val_files` | `$DATA_DIR/test.parquet` | `$DATA_DIR/test.parquet` | Test split for periodic validation |
| | `data.train_batch_size` | 64 | 512 | Total prompts sampled per RL iteration |
| | `data.max_prompt_length` | 512 | 1024 | Truncation limit for math question prompt |
| | `data.max_response_length`| 256 | 512 / 1024 | Max generated reasoning tokens |
| **Rollout (vLLM)** | `rollout.name` | `vllm` | `vllm` | High-throughput continuous batching |
| | `rollout.gpu_memory_utilization` | 0.4 | 0.6 | VRAM reserved for vLLM KV-cache pool |
| | `rollout.tensor_model_parallel_size`| 1 | 2 or 4 | Intra-node TP for generation |
| | `rollout.temperature` | 1.0 | 0.8 | Softmax sampling temperature |
| | `rollout.top_p` | 1.0 | 0.95 | Nucleus sampling cutoff |
| | `rollout.n` | 1 | 4 (GRPO) / 1 (PPO) | Rollouts generated per prompt |
| **Actor** | `actor.strategy` | `fsdp` | `fsdp` | PyTorch Fully Sharded Data Parallel |
| | `actor.optim.lr` | 1e-6 | 5e-7 | Learning rate (AdamW, $\beta_1=0.9, \beta_2=0.999$) |
| | `actor.ppo_mini_batch_size` | 16 | 64 | Minibatch size for SGD updates |
| | `actor.ppo_micro_batch_size`| 4 | 8 | Gradient accumulation slice per GPU |
| | `actor.ppo_epochs` | 1 | 1 | Training epochs per rollout batch |
| | `actor.clip_ratio` ($\epsilon$) | 0.2 | 0.2 | PPO surrogate clipping threshold |
| | `actor.entropy_coeff` | 0.001 | 0.001 | Entropy bonus to prevent premature collapse |
| **Critic** | `critic.strategy` | `fsdp` | `fsdp` | PyTorch FSDP |
| | `critic.optim.lr` | 5e-6 | 1e-6 | Critic LR (typically 2-5x Actor LR) |
| | `critic.clip_range_value` | 0.2 | 0.2 | Value function clipping bound |
| **Algorithm** | `algorithm.gamma` ($\gamma$) | 1.0 | 1.0 | Discount factor for finite horizon language tasks |
| | `algorithm.lam` ($\lambda$) | 0.95 | 0.95 | GAE parameter balancing bias and variance |
| | `algorithm.kl_ctrl.kl_coef` ($\beta$)| 0.05 | 0.02 | Per-token KL penalty coefficient |
| | `algorithm.kl_ctrl.type` | `fixed` | `adaptive` | Adaptive controller targets $D_{\text{target}}$ |
| **Trainer** | `trainer.total_epochs` | 3 | 5 | Total training passes over dataset |
| | `trainer.save_freq` | 10 | 50 | Checkpoint saving frequency (global steps) |
| | `trainer.test_freq` | 5 | 20 | Validation eval frequency on test split |

---

### 2.3 veRL Code Sandbox RLVR Production Recipe

From `code/chapter18_grpo/verl_code_rlvr/`:
- Launcher: `run_qwen_coder_ppo_single_gpu.sh`
- Data Prep: `prepare_data.py` (Eurus-2-RL-Data dataset)
- Verifier: `code_reward.py`

#### Key Architectural Requirements
1. **Chat Message Formatting**: Prompt column MUST be formatted as a structured list of dictionaries:
   `[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]`
   Passing raw plain strings causes `apply_chat_template` in Qwen tokenizers to discard content, emitting only 24 system tokens and producing persistent reward 0.
2. **Explicit Reward Wiring**: In veRL, custom Python rewards must be linked via launcher parameters:
   ```bash
   REWARD=(
       reward_model.enable=False
       custom_reward_function.path="code_reward.py"
       custom_reward_function.name=compute_score
   )
   ```
3. **Execution Sandbox Protocol**:
   The verifier extracts code blocks (`re.search(r"```(?:python)?
(.*?)```", re.DOTALL)`), executes stdin/stdout tests using `subprocess.run(..., timeout=10.0)`, and computes:
   $$\text{score} = \text{pass\_rate} = \frac{N_{\text{passed}}}{N_{\text{tests}}}$$
   Safety gate `HOMRL_ALLOW_UNSAFE_CODE_EXECUTION=1` must be explicitly verified before executing un-sandboxed code.

---

### 2.4 Financial Tool-Calling GRPO + TRL Recipe

From `chapter18_grpo/financial-tool-calling-grpo.md`:
- Model: Qwen2.5-0.5B-Instruct / 1.5B
- Trainer: Hugging Face `trl.GRPOTrainer`

#### Multi-Tiered Hierarchical Verifier
| Tier | Check | Scoring Formula | Failure Penalty |
| :--- | :--- | :--- | :--- |
| **Tier 1: Syntax** | JSON Valid Parsing | $r_{\text{format}} = 0.2$ if valid JSON else $0.0$ | Immediate return $0.0$ |
| **Tier 2: Schema** | Correct Tool Name & Allowed Arguments | $r_{\text{schema}} = 0.3$ if matches schema | Penalty $-0.5$ |
| **Tier 3: Execution** | Sandbox API Mock Call & Output Verification | $r_{\text{exec}} = 0.5$ if API returns expected data | $0.0$ |
| **Total Reward** | Weighted Sum | $r_{\text{total}} = r_{\text{format}} + r_{\text{schema}} + r_{\text{exec}} \in [0.0, 1.0]$ | |

---

### 2.5 Industrial Post-Training Architectures Across Frontier Labs

Extracted from Chapter 16 (`industrial-post-training.md`):

1. **DeepSeek (DeepSeek-R1-Zero / R1 / V3 / V3.2)**:
   - *R1-Zero*: Base model + GRPO + Rule-based verification (math/code) without SFT cold start. Emergent self-reflection ("Aha moment"). Language mixing and format defects.
   - *R1 Multi-Stage Pipeline*:
     1. Cold-start SFT (thousands of high-quality reasoning traces)
     2. Large-scale Reasoning RL (GRPO with rule verifiers)
     3. Rejection Sampling & SFT (600k reasoning samples + 200k non-reasoning alignment samples)
     4. Secondary RL (GRPO across all domains combining rule verifiers and preference RMs)
   - *Infrastructure*: DualPipe parallel pipeline, FP8 mixed precision, Multi-Token Prediction (MTP).
2. **Alibaba Qwen / Tongyi (Qwen2.5-Math, Qwen3, Qwen-Coder)**:
   - *GSPO*: Group Sequence Policy Optimization using geometric mean sequence ratios to eliminate MoE routing instability across 10,000-card GPU clusters.
   - *Thinking Mode Fusion*: Single unified checkpoint supporting both thinking (`/think`) and non-thinking inference via dynamic `thinking_budget`.
3. **Moonshot Kimi (Kimi k1.5)**:
   - *long2short RL*: Training-stage CoT compression. Replaces hard token limits with relative within-group length penalties:
     $$\lambda_i = 0.5 - \frac{\operatorname{len}(i) - l_{\min}}{l_{\max} - l_{\min}}$$
     Achieves 60.8 Pass@1 on AIME 2024 with average length compressed to 3,272 tokens.
4. **ByteDance Seed (Doubao, Seed-Thinking, VAPO)**:
   - *Dynamic Learnability Filtering*: Discards prompts with empirical pass rate $p 
otin [0.1, 0.9]$ where gradient variance collapses.
   - *VAPO*: Reintroduces Value Models for long CoT token-level credit assignment.
5. **MiniMax (MiniMax M1)**:
   - *CISPO*: Inverted importance sampling weight clipping to stabilize low-precision training on Lightning Attention across 512 H800 GPUs.
6. **Anthropic (Claude 2, 3, 3.5)**:
   - *Constitutional AI (CAI) + RLVR*: Safety aligned through natural language constitutional principles; reasoning tasks aligned through programmatic verifiers.
   - Context consistency loss $\mathcal{L}_{\text{context-consistency}}$ and attribution auditing $\mathcal{L}_{\text{audit}}$.
7. **OpenAI (o1, o3, InstructGPT)**:
   - Test-time compute scaling, PRM800K step-level supervision, and dense internal search over reasoning tokens.

---

### 2.6 Compute Estimation, FLOPs Modeling & Asynchronous Training Systems

#### Basic GPU-Hours Estimation Formula
$$\text{GPU-hours} \approx \frac{6 \cdot N_{\text{active}} \cdot N_{\text{tokens}}}{\text{GPU\_FLOPS} \cdot \text{MFU} \cdot 3600}$$

- $N_{\text{active}}$: Number of actively computing parameters per token (full model count for dense models; routed expert parameters for MoE).
- $N_{\text{tokens}}$: Total training tokens.
- Factor 6: 2 forward FLOPs + 4 backward FLOPs per parameter per token.
- $\text{MFU}$ (Model FLOPs Utilization): Realized compute utilization (typically 30%–50%).
- *Concrete Example*: Training a 7B dense model on 1B tokens on an NVIDIA H100 GPU (peak 989 TFLOPS BF16) at 40% MFU:
  $$\text{GPU-hours} = \frac{6 \times 7 \times 10^9 \times 10^9}{989 \times 10^{12} \times 0.40 \times 3600} \approx 29.5 \text{ GPU-hours}$$

#### Decomposed RL Step Cost Breakdown
$$C_{\text{RL-step}} = C_{\text{rollout}} + C_{\text{actor-update}} + C_{\text{ref-forward}} + C_{\text{reward}}$$

In typical 7B PPO configurations (batch size 512, rollout length 1024), **Rollout generation accounts for 50%–60% of total wall-clock runtime**.

#### Asynchronous RL & Staleness Ratio
In asynchronous systems (AReaL, LlamaRL), training updates lag behind rollouts. The policy staleness ratio is monitored:

$$\rho_t^{\text{stale}} = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{gen}}}(a_t \mid s_t)}$$

Trajectories are dropped if $|\rho_t^{\text{stale}} - 1| > \delta_{\text{stale}}$, preventing off-policy divergence.

---

# 3. Pathology Analysis: Failure Modes & Mitigation Mechanics

### 3.1 Length Bias & Verbosity Explosion

#### Failure Mechanism
1. **Reward Model Exploitation**: Learned RMs frequently exhibit positive correlation between text length and predicted reward ($\rho_{\text{reward, length}} = \operatorname{corr}(r_{\text{RM}}(x, y), |y|) > 0.6$). RMs mistake verbosity, extensive bullet points, and sycophantic polite framing for helpfulness.
2. **GRPO Standard Deviation Artifact**: In standard GRPO, normalizing by within-group standard deviation $\sigma_r$ rewards models for increasing within-group variance. The most accessible path to high variance is generating longer responses with higher token exploration.
3. **Hard Truncation Discontinuity**: Hard cutoffs at token length $L_{\max}$ penalize token $L_{\max}+1$ abruptly, introducing gradient shocks.

#### Mitigation Protocols
- **Dr.GRPO**: Eliminates standard deviation division ($\tilde{r}_i = r_i - \bar{r}$).
- **Length-Normalized Margins (SimPO)**: Divides log-probabilities by sequence length: $\frac{\beta}{|y|} \log \pi_\theta(y \mid x)$.
- **Smooth Overlong Reward Shaping (DAPO)**: Continuous quadratic penalty for tokens exceeding target budget.
- **Kimi k1.5 Relative Length Penalty**: Penalizes length relative to the shortest successful response within the same group:
  $$\lambda_i = 0.5 - \frac{\operatorname{len}(i) - l_{\min}}{l_{\max} - l_{\min}}$$

---

### 3.2 Reward Hacking & Goodhart's Law

> *"When a measure becomes a target, it ceases to be a good measure."* — Goodhart's Law

#### Anatomy of a Flawed Reward
In Chapter 15 (`extended-practice.md`), a deliberately flawed heuristic reward is analyzed:

$$R_{\text{bad}}(x, y) = 0.01 |y| + 0.5 \cdot \mathbb{1}[\text{has\_list}(y)] + 0.3 \cdot \mathbb{1}[\text{has\_polite\_phrase}(y)]$$

When optimized, the policy rapidly learns to output repetitive, nonsensical list items wrapped in excessive pleasantries (*"Certainly! I would be delighted to assist you with this wonderful inquiry: 1. A, 2. B, 3. C..."*). The reward curve increases monotonically while actual human win rate plummets from 65% to 12%.

#### Diagnostic Indicators of Reward Hacking
1. **Divergent Evaluation**: Training reward steadily rises while held-out benchmark accuracy (GSM8K, HumanEval) drops:
   $$\Delta_{\text{regression}} = \text{score}_{\text{RLHF}} - \text{score}_{\text{SFT}} < 0$$
2. **Length runaway**: Pearson correlation $\rho_{\text{reward, length}} \to 1.0$.
3. **KL Explosion**: $D_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}}) > 10.0$ nats.

#### Multi-Dimensional Reward Repair
To restore optimization stability, rewards must be decomposed into orthogonal components:

$$R_{\text{safe}}(x, y) = 0.4 R_{\text{helpful}} + 0.3 R_{\text{correct}} + 0.2 R_{\text{format}} - 0.05 R_{\text{length\_penalty}} - 0.05 R_{\text{repetition\_penalty}}$$

---

### 3.3 Variance Collapse & The Graduation Problem

#### Failure Mechanism
In GRPO and RLVR, advantage is computed exclusively within the group of $G$ rollouts for a single prompt.
- If a problem is too easy: all $G$ rollouts are correct ($r_1 = \dots = r_G = 1.0$).
- If a problem is too hard: all $G$ rollouts are incorrect ($r_1 = \dots = r_G = 0.0$).

In both cases, **within-group variance $\sigma_r = 0$, advantage $\hat{A}_i = 0$, and the gradient is identically zero**. Computing rollouts for these prompts wastes 60%–80% of cluster compute without providing any policy update.

#### Mitigation Protocols
- **Seed-Thinking Learnability Filter**: Pre-filters prompts by sampling $N=16$ rollouts from the current policy and discarding any prompt where pass rate $p 
otin [0.1, 0.9]$.
- **DAPO Dynamic Sampling**: Maintains a dynamic registry of problem mastery; prompts achieving pass rate $> 0.95$ are ejected from the active training pool.

---

### 3.4 KL Drift, Distribution Shift & Mode Collapse

#### Failure Dynamics
- **Unconstrained Policy Drift**: If KL coefficient $\beta$ is too small, $\pi_\theta$ drifts into low-probability regions of $\pi_{\text{ref}}$, resulting in gibberish text with high RM scores (adversarial examples against the RM).
- **Adaptive KL Controller**: In production PPO, $\beta$ is adjusted dynamically every step:
  $$\text{If } D_{\text{KL}} > 1.5 D_{\text{target}}: \quad \beta \leftarrow 1.5 \beta$$
  $$\text{If } D_{\text{KL}} < D_{\text{target}} / 1.5: \quad \beta \leftarrow \beta / 1.5$$
- **Mode Collapse in DPO**: When trained on deterministic datasets with small sample sizes, DPO drives implicit rewards to infinity, crushing entropy and causing the model to emit identical repetitive phrases. Regularized formulations like IPO directly mitigate this pathology.

---

### 3.5 Sycophancy, Over-Refusal & Format Degeneracy

- **Sycophancy**: Models falsely validate incorrect user premises to score higher on human preference models. Mitigated by explicit honesty penalties and RLVR verification.
- **Over-Refusal**: Safety fine-tuning induces extreme risk aversion, causing the model to refuse benign queries containing sensitive keywords (e.g. *"How do I kill a background python process?"*). Resolved in Constitutional AI by categorizing "sensitive but benign" training examples.
- **Format Collapse**: Reasoning models output valid structural tags (`<think>
</think>`) with empty or vacuous internal reasoning, satisfying format checkers while failing reasoning tasks. Resolved by pairing format rewards with strict length minimums and outcome verification.

---

# 4. Empirical Benchmarks, Case Studies & Comparative Matrices

### 4.1 EvalScope Benchmark Results (veRL Code RLVR)

Empirical evaluation from Chapter 18 (`verl-code-sandbox.md`), evaluating **Qwen2.5-7B-Instruct-1M** trained with veRL PPO on Eurus-2-RL-Data across 130 steps using EvalScope:

| Benchmark | Pre-RL Baseline | Post-RL (Step 130) | Net Delta ($\Delta$) | Analysis |
| :--- | :--- | :--- | :--- | :--- |
| **LiveCodeBench** | 0.50 (50.0%) | **0.53 (53.0%)** | **+3.0%** | Significant gain on competitive dynamic programming problems |
| **GSM8K** | 0.82 (82.0%) | **0.83 (83.0%)** | **+1.0%** | Cross-domain transfer from code logic to mathematical reasoning |
| **HumanEval** | 0.59 (59.0%) | 0.59 (59.0%) | 0.0% | Saturated on fixed standard library coding tasks |

---

### 4.2 Reasoning & Math Benchmarks

- **DAPO on AIME 2024**: Reached score **50.0** in **50% of the training steps** required by DeepSeek-R1, validating the compute efficiency of Clip-Higher and Dynamic Sampling.
- **Kimi k1.5 long2short RL on AIME 2024**: Achieved **Pass@1 of 60.8%** with an average response length compressed to **3,272 tokens**, significantly outperforming uncompressed baselines that required >8,000 tokens for equivalent accuracy.
- **PRM800K on MATH (Lightman et al., 2023)**: Process Reward Model (PRM) Best-of-$N$ search achieved **78.0%** on a representative subset of the MATH competition dataset, consistently outperforming Outcome Reward Models (ORM) at all compute budgets.
- **DPO Initial Loss Benchmark**: Evaluates to $-\ln(0.5) = \ln 2 \approx 0.693$ at step 0 across all model architectures, providing an immediate invariant check for pipeline sanity.

---

### 4.3 DPO vs PPO vs GRPO Architectural & Resource Comparison Matrix

| Dimension | PPO-RLHF | DPO | SimPO | GRPO | GSPO |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Online / Offline** | Online | Offline | Offline | Online | Online |
| **Concurrent Active Models** | 4 (Actor, Critic, Ref, RM)| 2 (Policy, Ref) | **1 (Policy only)** | 2 (Actor, Ref) | 2 (Actor, Ref) |
| **Critic Model Required?** | **Yes** (Equal size to Actor) | No | No | **No** (Group Mean) | **No** (Group Mean) |
| **Reward Model Required?** | **Yes** (Neural RM) | No (Implicit) | No (Implicit) | Optional (Rule/Verif)| Optional (Rule/Verif)|
| **VRAM Footprint vs SFT** | **3.0x – 4.0x** | ~1.5x – 2.0x | **1.0x** | ~1.5x – 2.0x | ~1.5x – 2.0x |
| **Rollout Throughput Bound** | Critical Bottleneck | None (Pre-computed) | None | Critical Bottleneck | Critical Bottleneck |
| **Exploration Capability** | High (Online sampling) | None (Dataset bounded) | None | High (Online sampling) | High (Online sampling) |
| **Credit Assignment** | Dense (Token GAE) | Pairwise Sequence | Pairwise Sequence | Sparse Outcome / Token | Geometric Mean Seq |
| **MoE Cluster Stability** | Moderate | High | High | Low (Ratio variance) | **Extremely High** |

---

### 4.4 Complete Algorithm Taxonomy & Decision Tree

```text
Alignment & Post-Training Task
├── Is the Task Verifiable via Deterministic Rules/Code/Formal Logic?
│   ├── YES (Mathematics, Programming, Formal Logic, Structured API Tool Calling)
│   │   ├── Online Exploration Feasible?
│   │   │   ├── YES → Verifiable Rewards (RLVR)
│   │   │   │   ├── Dense Architecture → GRPO / DAPO (with Token-level credit assignment)
│   │   │   │   ├── Large MoE Cluster (e.g. Qwen3) → GSPO (Geometric mean ratio) + Dr.GRPO (No std)
│   │   │   │   ├── Ultra-long CoT Reasoning → VAPO (Value-augmented advantage)
│   │   │   │   └── Zero SFT Cold Start → DeepSeek-R1-Zero Pure RL Loop
│   │   │   └── NO (Fixed offline trajectories) → On-Policy Distillation (OPD / GKD)
│   │   └── Step-level Search Desired at Inference?
│   │       ├── Search Tree Reusable → MCTS (PUCT) / Beam Search + PRM (PRM800K / Lean4)
│   │       └── Parallel Candidate Sampling → Best-of-N + PaCoRe / GenRM Coordinator
│   └── NO (Open-ended Dialogue, Creative Writing, Safety, Ethics, Tone)
│       ├── Pairwise Preference Dataset Available?
│       │   ├── High Compute Budget & Online Exploration Desired → Classic PPO-RLHF
│       │   ├── Constrained Compute & Fast Iteration Desired → DPO
│       │   ├── Single-Card GPU Memory Limit (No Ref Model) → SimPO
│       │   └── Small Dataset (<1000 pairs) & Overfitting Risk → IPO
│       ├── Unpaired Thumbs Up/Down Binary User Feedback → KTO
│       └── Principle-Guided Safety Alignment → Constitutional AI (SL-CAI + RLAIF HHH Loop)
```

---
*End of Extraction Manual. Compiled directly from source chapters 15 through 21.*

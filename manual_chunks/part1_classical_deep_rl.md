# Classical & Deep Reinforcement Learning: Comprehensive Mathematical, Algorithmic, and Engineering Reference Manual

> **Scope**: This document provides an exhaustive, mathematically rigorous, code-level reference synthesis of Classical and Deep Reinforcement Learning covering Chapters 01, 03, 07, 08, 09, 10, and 11 of the *Hands-on Modern RL* curriculum across 42 foundational markdown manuscripts.
> 
> **Content Guarantee**: Every mathematical formula, Bellman equation, loss formulation, algorithm derivation, proof, core table, hyperparameter specification, failure mode, bias-variance tradeoff, and engineering diagnostic takeaway has been extracted and systematically integrated without approximation or omission.

---

## Table of Contents

1. [Chapter 01: CartPole Foundations & Diagnostic Metrics](#1-chapter-01-cartpole-foundations--diagnostic-metrics)
   - 1.1 [CartPole Environment Specification & Physical Dynamics](#11-cartpole-environment-specification--physical-dynamics)
   - 1.2 [The Reinforcement Learning Core Control Loop](#12-the-reinforcement-learning-core-control-loop)
   - 1.3 [Actor-Critic Architecture & PPO Pipeline Inside CartPole](#13-actor-critic-architecture--ppo-pipeline-inside-cartpole)
   - 1.4 [Four Core Diagnostic Metrics for Policy Health](#14-four-core-diagnostic-metrics-for-policy-health)
   - 1.5 [Diagnostic Evaluation Protocol & Debugging Order](#15-diagnostic-evaluation-protocol--debugging-order)
   - 1.6 [Reproducible Benchmark Configurations & Measured Settings](#16-reproducible-benchmark-configurations--measured-settings)

2. [Chapter 03: Markov Decision Processes & Tabular Foundations](#2-chapter-03-markov-decision-processes--tabular-foundations)
   - 2.1 [Multi-Armed Bandits & Exploration-Exploitation Dilemma](#21-multi-armed-bandits--exploration-exploitation-dilemma)
   - 2.2 [Markov Decision Process (MDP) Formal Tuple & Dynamics](#22-markov-decision-process-mdp-formal-tuple--dynamics)
   - 2.3 [Discounted Cumulative Return & Trajectory Horizon](#23-discounted-cumulative-return--trajectory-horizon)
   - 2.4 [Policies, State-Value $V(s)$, and Action-Value $Q(s, a)$](#24-policies-state-value-vs-and-action-value-qsa)
   - 2.5 [The Advantage Function $A(s, a)$](#25-the-advantage-function-asa)
   - 2.6 [Bellman Expectation Equations: Derivation, Decomposition, & Matrix Form](#26-bellman-expectation-equations-derivation-decomposition--matrix-form)
   - 2.7 [Bellman Optimality Equations & Fixed-Point Theory](#27-bellman-optimality-equations--fixed-point-theory)
   - 2.8 [Dynamic Programming: Policy Iteration & Value Iteration](#28-dynamic-programming-policy-iteration--value-iteration)
   - 2.9 [Q-Learning: Off-Policy TD Control & Exploratory Policies](#29-q-learning-off-policy-td-control--exploratory-policies)
   - 2.10 [The Triad: Dynamic Programming vs. Monte Carlo vs. Temporal Difference](#210-the-triad-dynamic-programming-vs-monte-carlo-vs-temporal-difference)
   - 2.11 [Reinforcement Learning Taxonomy & Data Regimes](#211-reinforcement-learning-taxonomy--data-regimes)
   - 2.12 [Reward Design, Potential Shaping, & Modern Alignment Models](#212-reward-design-potential-shaping--modern-alignment-models)

3. [Chapter 07: Deep Q-Networks (DQN)](#3-chapter-07-deep-q-networks-dqn)
   - 3.1 [Transition from Tabular Q-Learning to Function Approximation](#31-transition-from-tabular-q-learning-to-function-approximation)
   - 3.2 [Instability Phenomena: Moving Targets & Correlated Transitions](#32-instability-phenomena-moving-targets--correlated-transitions)
   - 3.3 [The Two Pillars of DQN: Experience Replay & Target Networks](#33-the-two-pillars-of-dqn-experience-replay--target-networks)
   - 3.4 [Loss Formulations: MSE vs. Huber / Smooth L1](#34-loss-formulations-mse-vs-huber--smooth-l1)
   - 3.5 [Complete Deep Q-Network Algorithm & Control Flow](#35-complete-deep-q-network-algorithm--control-flow)
   - 3.6 [The DQN Improvement Family (Rainbow Innovations)](#36-the-dqn-improvement-family-rainbow-innovations)
   - 3.7 [Continuous Observation Benchmark: LunarLander-v2](#37-continuous-observation-benchmark-lunarlander-v2)
   - 3.8 [Visual Game Projects: Atari Preprocessing, Architecture, & Task Frontiers](#38-visual-game-projects-atari-preprocessing-architecture--task-frontiers)

4. [Chapter 08: Policy Gradients & REINFORCE](#4-chapter-08-policy-gradients--reinforce)
   - 4.1 [Why Policy Gradients: Value-Based Breakdown & Action Distribution](#41-why-policy-gradients-value-based-breakdown--action-distribution)
   - 4.2 [The Policy Objective Formulation $J(\theta)$](#42-the-policy-objective-formulation-jtheta)
   - 4.3 [The Policy Gradient Theorem: Analytic Derivation & Score Function Trick](#43-the-policy-gradient-theorem-analytic-derivation--score-function-trick)
   - 4.4 [Causality, Temporal Independence, & Reward-to-Go](#44-causality-temporal-independence--reward-to-go)
   - 4.5 [Baseline Invariance Proof & Variance Reduction](#45-baseline-invariance-proof--variance-reduction)
   - 4.6 [The REINFORCE Algorithm & Trajectory Rollout Loop](#46-the-reinforce-algorithm--trajectory-rollout-loop)
   - 4.7 [Discrete vs. Continuous Policy Formulations](#47-discrete-vs-continuous-policy-formulations)
   - 4.8 [Experimental Validations: Bandit Oscillations & CartPole Baselines](#48-experimental-validations-bandit-oscillations--cartpole-baselines)

5. [Chapter 09: Actor-Critic Architectures](#5-chapter-09-actor-critic-architectures)
   - 5.1 [From Monte Carlo Policy Gradient to Actor-Critic](#51-from-monte-carlo-policy-gradient-to-actor-critic)
   - 5.2 [The Actor-Critic Bias-Variance Tradeoff](#52-the-actor-critic-bias-variance-tradeoff)
   - 5.3 [Advantage Function Estimation via One-Step TD Error](#53-advantage-function-estimation-via-one-step-td-error)
   - 5.4 [Network Architecture: Shared Backbone vs. Disjoint Parameterization](#54-network-architecture-shared-backbone-vs-disjoint-parameterization)
   - 5.5 [Critic Training Mechanics: DP vs. MC vs. TD Walkthrough](#55-critic-training-mechanics-dp-vs-mc-vs-td-walkthrough)
   - 5.6 [Continuous Action Control: Gaussian Policy Parameterization](#56-continuous-action-control-gaussian-policy-parameterization)
   - 5.7 [Complex Control Benchmark: Pendulum-v1 & BipedalWalker-v3](#57-complex-control-benchmark-pendulum-v1--bipedalwalker-v3)
   - 5.8 [Large-Scale Frontiers: AlphaGo, AlphaStar, SAC, & NVIDIA Isaac Lab](#58-large-scale-frontiers-alphago-alphastar-sac--nvidia-isaac-lab)

6. [Chapter 10: Proximal Policy Optimization (PPO)](#6-chapter-10-proximal-policy-optimization-ppo)
   - 6.1 [Vanilla Policy Gradient Failure: Step Size Sensitivity & Collapse](#61-vanilla-policy-gradient-failure-step-size-sensitivity--collapse)
   - 6.2 [Importance Sampling & Off-Policy Trajectory Reuse](#62-importance-sampling--off-policy-trajectory-reuse)
   - 6.3 [Trust Region Policy Optimization (TRPO) & Second-Order Constraints](#63-trust-region-policy-optimization-trpo--second-order-constraints)
   - 6.4 [PPO Clipped Surrogate Objective: Full Analytical Derivation](#64-ppo-clipped-surrogate-objective-full-analytical-derivation)
   - 6.5 [Generalized Advantage Estimation (GAE): Derivation & Recursive Form](#65-generalized-advantage-estimation-gae-derivation--recursive-form)
   - 6.6 [The Full Composite PPO Loss Function](#66-the-full-composite-ppo-loss-function)
   - 6.7 [Complete PPO Training Algorithm & Engineering Implementations](#67-complete-ppo-training-algorithm--engineering-implementations)
   - 6.8 [Complex Locomotion: BipedalWalker-v3 PPO Benchmark](#68-complex-locomotion-bipedalwalker-v3-ppo-benchmark)
   - 6.9 [Comprehensive Game Project Evaluation Benchmark](#69-comprehensive-game-project-evaluation-benchmark)
   - 6.10 [Long-Horizon Tasks, Exploration Frontiers, & Planning Paradigms](#610-long-horizon-tasks-exploration-frontiers--planning-paradigms)

7. [Chapter 11: Continuous Control & Advanced Architectures](#7-chapter-11-continuous-control--advanced-architectures)
   - 7.1 [Deterministic Policy Gradients (DPG) & Deep DPG (DDPG)](#71-deterministic-policy-gradients-dpg--deep-dpg-ddpg)
   - 7.2 [DDPG Failure Modes: Q-Overestimation & Brittleness](#72-ddpg-failure-modes-q-overestimation--brittleness)
   - 7.3 [Twin Delayed Deep Deterministic Policy Gradient (TD3)](#73-twin-delayed-deep-deterministic-policy-gradient-td3)
   - 7.4 [Soft Actor-Critic (SAC): Maximum Entropy Reinforcement Learning](#74-soft-actor-critic-sac-maximum-entropy-reinforcement-learning)
   - 7.5 [Comparison: DDPG vs. TD3 vs. SAC](#75-comparison-ddpg-vs-td3-vs-sac)
   - 7.6 [Model-Based Reinforcement Learning: Paradigms & Compounding Error](#76-model-based-reinforcement-learning-paradigms--compounding-error)
   - 7.7 [Dyna-Q, PETS, and Model-Based Policy Optimization (MBPO)](#77-dyna-q-pets-and-model-based-policy-optimization-mbpo)
   - 7.8 [Search & World Models: AlphaZero, MuZero, and Dreamer V3](#78-search--world-models-alphazero-muzero-and-dreamer-v3)

---

# 1. Chapter 01: CartPole Foundations & Diagnostic Metrics

### Sources
- `docs/en/chapter01_cartpole/principles.md`
- `docs/en/chapter01_cartpole/metrics.md`
- `docs/en/chapter01_cartpole/training.md`

---

## 1.1 CartPole Environment Specification & Physical Dynamics

CartPole (`CartPole-v1`) is the canonical inverted pendulum benchmark modeled in Gymnasium. An unactuated pole is attached by an un-actuated joint to a cart moving along a frictionless track. The system state is 4-dimensional continuous space, while the control is 1-dimensional discrete space.

### State Space $\mathcal{S}$
The observation vector $s_t \in \mathbb{R}^4$ is structured as:

| Index | State Feature | Notation | Range in Observation Space | Physical Termination Threshold |
| :---: | :--- | :---: | :---: | :---: |
| 0 | Cart Position | $x$ | $[-4.8, 4.8]$ | $|x| > 2.4$ |
| 1 | Cart Velocity | $\dot{x}$ | $(-\infty, \infty)$ | None |
| 2 | Pole Angle | $\theta$ | $[-0.4189, 0.4189]\text{ rad}$ (approx $\pm 24^\circ$) | $|\theta| > 0.2095\text{ rad}$ (approx $\pm 12^\circ$) |
| 3 | Pole Angular Velocity | $\dot{\theta}$ | $(-\infty, \infty)$ | None |

### Action Space $\mathcal{A}$
The agent interacts via discrete horizontal force applications:

| Action $a_t$ | Physical Meaning | Applied Force Vector |
| :---: | :--- | :--- |
| `0` | Push cart to the Left | $-10.0\text{ N}$ |
| `1` | Push cart to the Right | $+10.0\text{ N}$ |

### Reward Dynamics & Termination Criteria
- **Step Reward**: $r_t = +1.0$ for every timestep the system remains within physical stability constraints.
- **Episode Termination Conditions (`terminated = True`)**:
  1. Pole angle condition: $|\theta| > 12^\circ$ ($0.2095\text{ rad}$).
  2. Cart position condition: $|x| > 2.4$ (cart reaches track boundaries).
- **Episode Truncation Condition (`truncated = True`)**:
  - Step limit reached: $t = 500$ timesteps in `CartPole-v1` (maximum theoretical undiscounted return $G = 500.0$).

---

## 1.2 The Reinforcement Learning Core Control Loop

The fundamental closed-loop interaction between agent and environment proceeds in discrete timesteps:
1. Environment provides initial state $s_0 \sim \rho_0(s)$.
2. At each timestep $t$:
   - Agent observes state $s_t \in \mathcal{S}$.
   - Agent policy $\pi(a_t \mid s_t)$ samples an action $a_t \in \mathcal{A}$.
   - Environment transitions to next state $s_{t+1} \sim P(s_{t+1} \mid s_t, a_t)$ and emits scalar reward $r_t = R(s_t, a_t, s_{t+1})$.
   - Environment evaluates Boolean flags `terminated` and `truncated`.
3. Agent uses the sequence of transitions $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T)$ to optimize policy parameters $\theta$ to maximize expected cumulative return:
   $$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\sum_{t=0}^{T} \gamma^t r_t\right]$$

---

## 1.3 Actor-Critic Architecture & PPO Pipeline Inside CartPole

While high-level frameworks like Stable-Baselines3 encapsulate training within `model.learn()`, the underlying pure PyTorch implementation executes a four-stage decoupled pipeline:

```
+------------------------------------------------------------------------------------+
| 1. Rollout Collection: Policy interacts with Gym CartPole-v1 for T_rollout steps    |
|    Records: s_t, a_t, r_t, log pi_old(a_t | s_t), V(s_t), terminated_t             |
+------------------------------------------------------------------------------------+
                                         |
                                         v
+------------------------------------------------------------------------------------+
| 2. GAE Computation: Backward temporal pass over rollout transitions                |
|    delta_t = r_t + gamma * V(s_{t+1}) * (1 - term) - V(s_t)                       |
|    A_t^{GAE} = delta_t + gamma * lambda * (1 - term) * A_{t+1}^{GAE}              |
|    Target Returns: R_t^{targ} = A_t^{GAE} + V(s_t)                                 |
+------------------------------------------------------------------------------------+
                                         |
                                         v
+------------------------------------------------------------------------------------+
| 3. Mini-Batch SGD Optimization: K epochs over mini-batches of size B               |
|    Ratio: r_t(theta) = pi_theta(a_t | s_t) / pi_old(a_t | s_t)                     |
|    L^{CLIP} = min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)                        |
|    L^{VF} = (V_theta(s_t) - R_t^{targ})^2                                          |
|    L^{ENT} = - sum_a pi_theta(a | s_t) * log pi_theta(a | s_t)                     |
|    Total Loss: L = - L^{CLIP} + c_1 * L^{VF} - c_2 * L^{ENT}                       |
+------------------------------------------------------------------------------------+
                                         |
                                         v
+------------------------------------------------------------------------------------+
| 4. Health Diagnostics Logging: Track Entropy, Value Loss, Approx KL, Clip Fraction |
+------------------------------------------------------------------------------------+
```

### Network Topology
- **Actor Network**: MLP mapping $\mathbb{R}^4 \to \mathbb{R}^2$ logits. Architecture: `Linear(4, 64) -> Tanh -> Linear(64, 64) -> Tanh -> Linear(64, 2) -> Categorical Softmax`.
- **Critic Network**: MLP mapping $\mathbb{R}^4 \to \mathbb{R}^1$ scalar value. Architecture: `Linear(4, 64) -> Tanh -> Linear(64, 64) -> Tanh -> Linear(64, 1)`.

---

## 1.4 Four Core Diagnostic Metrics for Policy Health

Reward curves alone cannot detect policy pathologies. Four distinct diagnostic signals must be monitored simultaneously:

### 1. Policy Entropy ($H$)
Measures the stochasticity / uncertainty of the policy distribution over action space:
$$H(\pi_\theta(\cdot \mid s_t)) = - \sum_{a \in \mathcal{A}} \pi_\theta(a \mid s_t) \log \pi_\theta(a \mid s_t)$$
- **Theoretical Bounds**: For $|\mathcal{A}| = 2$, maximum entropy is $\ln 2 \approx 0.6931\text{ nats}$ (uniform random exploration). As policy becomes fully deterministic, $H \to 0.0$.
- **Healthy Progression**: Begins near $0.69$, exhibiting a slow, controlled decline as the agent masters balancing.
- **Pathology Indicator**: A rapid, premature drop to near $0.0$ indicates premature convergence to a suboptimal sub-policy (e.g., repeatedly pushing left until hitting the wall).

### 2. Value Function Loss ($L_V$) & Explained Variance ($EV$)
Measures the critic network's accuracy in forecasting long-term returns:
$$L_V = \frac{1}{|B|} \sum_{i \in B} \left( V_\phi(s_i) - \hat{G}_i \right)^2$$
Where $\hat{G}_i = \hat{A}_i + V_{\text{old}}(s_i)$ is the empirical target return.
The predictive quality is formalized via Explained Variance:
$$EV = 1 - \frac{\text{Var}(y - V_\phi(s))}{\text{Var}(y)}$$
- $EV \approx 1.0$: Critic perfectly predicts empirical variance of returns.
- $EV \approx 0.0$: Critic is no better than predicting a constant mean return.
- $EV < 0.0$: Severe critic collapse; predictions worsen variance (overfitting or non-stationary targets).

### 3. Approximate KL Divergence ($D_{KL}$)
Monitors the step size taken in policy probability space between iterations $\theta_{\text{old}}$ and $\theta_{\text{new}}$:
$$D_{KL}(\pi_{\text{old}} \parallel \pi_\theta) \approx \frac{1}{|B|} \sum_{i \in B} \left[ \left( r_i(\theta) - 1 \right) - \log r_i(\theta) \right]$$
Alternatively, the variance-based estimator:
$$D_{KL} \approx \frac{1}{2 |B|} \sum_{i \in B} \left( \log r_i(\theta) \right)^2$$
- **Target Threshold**: Typically maintained in the range $[0.01, 0.03]$.
- **Pathology Indicator**: Spikes above $0.05$ indicate violent policy updates that destabilize training and violate the local trust region assumption.

### 4. Clip Fraction ($\text{clipfrac}$)
Measures the proportion of sample transitions where importance sampling ratio $r_i(\theta)$ triggers clipping bounds $[1-\epsilon, 1+\epsilon]$:
$$\text{clipfrac} = \frac{1}{|B|} \sum_{i \in B} \mathbf{1}\left[ |r_i(\theta) - 1| > \epsilon \right]$$
- **Target Threshold**: Healthy clipping operates between $0.05$ and $0.20$ ($5\% - 20\%$).
- **Pathology Indicator**: If $\text{clipfrac} > 0.40$, policy is updating too fast, truncating gradients on nearly half the data. If $\text{clipfrac} \approx 0.0$, the clipping constraint is completely inactive (learning rate may be too low).

---

## 1.5 Diagnostic Evaluation Protocol & Debugging Order

When an RL agent experiences performance degradation or divergence, follow the strict minimal diagnostic triage sequence:

```
[Level 1: Evaluation vs. Training Return Gap]
  |-- Check: Is eval return >> training return?
  |   +-- Normal: Exploration noise degrades training returns; eval is deterministic.
  |-- Check: Did eval return crash suddenly?
      +-- Yes -> Investigate Level 2.
  v
[Level 2: Policy Entropy Check]
  |-- Did entropy crash to 0 in early iterations?
  |   +-- Diagnosis: Premature policy collapse. Action space collapsed before finding reward signal.
  |   +-- Remedy: Increase entropy coefficient c_2, reduce learning rate.
  v
[Level 3: Policy Shift Diagnostics (Approx KL & Clip Fraction)]
  |-- Did Approx KL spike (> 0.05) or clipfrac exceed 30%?
  |   +-- Diagnosis: Gradient steps are too aggressive; policy fell off a performance cliff.
  |   +-- Remedy: Lower learning rate, reduce mini-batch epochs, decrease clipping parameter epsilon.
  v
[Level 4: Critic Quality (Value Loss & Explained Variance)]
  |-- Is Explained Variance EV < 0?
      +-- Diagnosis: Critic failed to fit value manifold; advantage estimates are pure noise.
      +-- Remedy: Adjust critic learning rate, increase rollout buffer size, tune GAE lambda.
```

---

## 1.6 Reproducible Benchmark Configurations & Measured Settings

The baseline runs documented in the repository establish the following validated configurations:

| Parameter Category | Configuration Attribute | Measured Value / Setting |
| :--- | :--- | :--- |
| **System** | Python / PyTorch / Gym / NumPy | Python 3.12.13, PyTorch 2.13.0, Gymnasium 1.3.0, NumPy 2.5.2 |
| **Environment** | Identifier / Max Horizon | `CartPole-v1` / 500 timesteps |
| **Algorithm** | Implementation | Decoupled Pure PyTorch PPO |
| **Random Seed** | Deterministic Seed | `42` |
| **Compute Budget** | Total Steps / Iterations | 40 iterations $\times$ 2,048 steps = 81,920 environment steps |
| **Rollout Length** | Steps per Rollout ($T_{\text{rollout}}$) | 2,048 steps |
| **Optimization** | Optimizer / Learning Rate | Adam / $\alpha = 3 \times 10^{-4}$ |
| **PPO Hyperparameters** | $\epsilon$ (clip) / GAE $\lambda$ / Discount $\gamma$ | $\epsilon = 0.2$, $\lambda = 0.95$, $\gamma = 0.99$ |
| **Coefficients** | Value $c_1$ / Entropy $c_2$ | $c_1 = 0.5$, $c_2 = 0.0$ (or $0.01$) |
| **Mini-batching** | Epochs ($K$) / Batch Size ($B$) | 10 epochs / 64 transitions per mini-batch |
| **Evaluation** | Protocol | 20 independent episodes, greedy deterministic policy ($\arg\max \pi$) |
| **Performance Criterion** | Mean Evaluation Return | $500.0 \pm 0.0$ (Environment fully solved) |


# 2. Chapter 03: Markov Decision Processes & Tabular Foundations

### Sources
- `docs/en/chapter03_mdp/bandit.md`
- `docs/en/chapter03_mdp/mdp.md`
- `docs/en/chapter03_mdp/policy-value.md`
- `docs/en/chapter03_mdp/value-bellman.md`
- `docs/en/chapter03_mdp/value-q.md`
- `docs/en/chapter03_mdp/value-experiment.md`
- `docs/en/chapter03_mdp/dp-mc-td.md`
- `docs/en/chapter03_mdp/algorithm-taxonomy.md`
- `docs/en/chapter03_mdp/reward-design.md`

---

## 2.1 Multi-Armed Bandits & Exploration-Exploitation Dilemma

The multi-armed bandit formalizes decision-making under uncertainty in a single state ($|\mathcal{S}| = 1$). An agent selects from $K$ actions (arms) $a \in \{1, \ldots, K\}$, each delivering reward drawn from an unknown stationary distribution with mean $\mu_a = \mathbb{E}[R_a]$.

### Canonical Two-Armed Slot Machine Formulation
Consider two Bernoulli slot machines with winning payout $+1$ and losing payout $-1$:
- Arm A true win probability: $p_A = 0.60$
- Arm B true win probability: $p_B = 0.40$
The expected reward for arm $a$ is:
$$\mathbb{E}[R_a] = p_a \cdot (+1) + (1 - p_a) \cdot (-1) = 2p_a - 1$$
- $\mathbb{E}[R_A] = 2(0.60) - 1 = +0.20$ (positive expectation)
- $\mathbb{E}[R_B] = 2(0.40) - 1 = -0.20$ (negative expectation)

### Policy Comparison over $T = 100$ Rounds
1. **Uniform Random Policy**:
   $$P(A) = 0.5, \quad P(B) = 0.5 \implies \mathbb{E}[R_{\text{step}}] = 0.5(0.2) + 0.5(-0.2) = 0.0 \implies \mathbb{E}[R_{100}] = 0.0$$
2. **Oracle Optimal Policy (Always A)**:
   $$P(A) = 1.0 \implies \mathbb{E}[R_{100}] = 100 \times (+0.20) = +20.0$$
3. **Explore-Then-Commit (ETC)**:
   - *Phase 1 (Exploration)*: Pull Arm A 10 times, Arm B 10 times (20 steps).
   - *Phase 2 (Commitment)*: Identify $\hat{a} = \arg\max(\hat{p}_A, \hat{p}_B)$ and pull $\hat{a}$ for remaining 80 steps.

#### Exact Finite-Sample Error Derivation
The empirical win count $N_A \sim \text{Binomial}(10, 0.6)$ and $N_B \sim \text{Binomial}(10, 0.4)$. The probability of committing to the inferior Arm B is:
$$P(\text{Mistake}) = P(N_A < N_B) + \frac{1}{2} P(N_A = N_B) = 12.8\%$$
The expected total return under ETC is:
$$\mathbb{E}[R_{100}] = (1 - P(\text{Mistake})) \cdot [0 + 80(+0.2)] + P(\text{Mistake}) \cdot [0 + 80(-0.2)]$$
$$\mathbb{E}[R_{100}] = 0.872 \times 16.0 + 0.128 \times (-16.0) \approx 13.952 - 2.048 \approx 11.904$$

### Thompson Sampling (Posterior Sampling)
Maintains Beta distribution priors over win probabilities: $\theta_a \sim \text{Beta}(\alpha_a, \beta_a)$ with uniform prior $\alpha_a = 1, \beta_a = 1$:
1. Sample candidate success probabilities: $\tilde{p}_a \sim \text{Beta}(\alpha_a, \beta_a)$ for each arm.
2. Select action $a_t = \arg\max_a \tilde{p}_a$.
3. Observe outcome $r_t \in \{0, 1\}$.
4. Update Bayesian posterior:
   $$\text{If win } (r_t = 1): \alpha_{a_t} \leftarrow \alpha_{a_t} + 1; \quad \text{If loss } (r_t = 0): \beta_{a_t} \leftarrow \beta_{a_t} + 1$$
Thompson Sampling naturally drives exploration through posterior variance: highly uncertain arms exhibit wide Beta distributions, producing occasionally large sampled $\tilde{p}_a$ values.

### Regret Bounds
Cumulative regret measures performance loss relative to the oracle:
$$R_T = T \cdot \mu^* - \sum_{t=1}^T \mathbb{E}[R_{a_t}] = \sum_{a: \mu_a < \mu^*} \Delta_a \mathbb{E}[N_a(T)]$$
where $\Delta_a = \mu^* - \mu_a$. By the Lai-Robbins Lower Bound, no consistent algorithm can achieve asymptotic regret lower than:
$$\liminf_{T \to \infty} \frac{R_T}{\ln T} \ge \sum_{a: \mu_a < \mu^*} \frac{\Delta_a}{D_{KL}(P_a \parallel P^*)}$$

---

## 2.2 Markov Decision Process (MDP) Formal Tuple & Dynamics

The formal mathematical foundation of sequential decision making is the MDP 5-tuple:
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$$

| Component | Mathematical Definition | Physical Meaning |
| :---: | :--- | :--- |
| $\mathcal{S}$ | State space (discrete or continuous manifold) | Complete configuration of environment |
| $\mathcal{A}$ | Action space (discrete or continuous manifold) | Set of available decisions |
| $P$ | Transition probability kernel $P(s' \mid s, a) = \mathbb{P}(S_{t+1}=s' \mid S_t=s, A_t=a)$ | Environmental conditional physics |
| $R$ | Reward function $R(s, a) = \mathbb{E}[R_{t+1} \mid S_t=s, A_t=a]$ | Immediate scalar reinforcement signal |
| $\gamma$ | Discount factor $\gamma \in [0, 1)$ | Geometric temporal penalty factor |

### The Markov Property
A state $S_t$ satisfies the Markov property if and only if transition dynamics depend strictly on the present state and action, exhibiting complete conditional independence from past history:
$$\mathbb{P}(S_{t+1}=s_{t+1}, R_{t+1}=r_{t+1} \mid S_t=s_t, A_t=a_t, S_{t-1}=s_{t-1}, A_{t-1}=a_{t-1}, \ldots, S_0=s_0, A_0=a_0)$$
$$= \mathbb{P}(S_{t+1}=s_{t+1}, R_{t+1}=r_{t+1} \mid S_t=s_t, A_t=a_t)$$

---

## 2.3 Discounted Cumulative Return & Trajectory Horizon

The performance of an agent along trajectory $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ is quantified by the discounted cumulative return:
$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k} = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots$$

### The Fundamental Bellman Recursion of Returns
$$G_t = r_t + \gamma \left( r_{t+1} + \gamma r_{t+2} + \cdots \right) = r_t + \gamma G_{t+1}$$

### Analytical Roles of the Discount Factor $\gamma$
1. **Mathematical Convergence**: For infinite-horizon MDPs with bounded rewards $|r_t| \le R_{\max}$, $\gamma < 1$ guarantees finite return:
   $$G_t \le \sum_{k=0}^{\infty} \gamma^k R_{\max} = \frac{R_{\max}}{1 - \gamma} < \infty$$
2. **Effective Horizon**: The effective planning horizon is approximated by $\frac{1}{1 - \gamma}$ timesteps (e.g., $\gamma=0.9 \implies 10\text{ steps}$, $\gamma=0.99 \implies 100\text{ steps}$).
3. **Uncertainty Quantification**: Far-future events carry compounding environmental stochasticity; discounting reflects decreasing predictive confidence.

---

## 2.4 Policies, State-Value $V(s)$, and Action-Value $Q(s, a)$

### Policies
- **Deterministic Policy**: A mapping $\pi: \mathcal{S} \to \mathcal{A}$ where $a = \pi(s)$.
- **Stochastic Policy**: A conditional probability distribution $\pi(a \mid s) = \mathbb{P}(A_t = a \mid S_t = s)$ such that $\sum_{a \in \mathcal{A}} \pi(a \mid s) = 1$.

### State-Value Function $V^\pi(s)$
Measures expected return starting from state $s$ following policy $\pi$:
$$V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid S_t = s \right] = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k} \;\middle|\; S_t = s \right]$$

### Action-Value Function $Q^\pi(s, a)$
Measures expected return starting from state $s$, executing arbitrary action $a$, and thereafter following policy $\pi$:
$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t \mid S_t = s, A_t = a \right] = \mathbb{E}_\pi \left[ r_t + \gamma \sum_{k=0}^{\infty} \gamma^k r_{t+1+k} \;\middle|\; S_t = s, A_t = a \right]$$

### The Bridge Relationship Between $V$ and $Q$
$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) Q^\pi(s, a)$$

---

## 2.5 The Advantage Function $A(s, a)$

The advantage measures the relative quality of an action against the policy's baseline performance in that state:
$$A^\pi(s, a) = Q^\pi(s, a) - V^\pi(s)$$
- If $A^\pi(s, a) > 0$: Action $a$ performs better than the average action under $\pi$.
- If $A^\pi(s, a) < 0$: Action $a$ performs worse than average.
- Under expectation across the policy, the advantage is zero:
  $$\mathbb{E}_{a \sim \pi(\cdot \mid s)} \left[ A^\pi(s, a) \right] = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left( Q^\pi(s, a) - V^\pi(s) \right) = V^\pi(s) - V^\pi(s) = 0$$

---

## 2.6 Bellman Expectation Equations: Derivation, Decomposition, & Matrix Form

### Analytic Derivation of the Bellman Expectation Equation for $V^\pi(s)$
Expanding the expectation definition via law of total expectation:
$$V^\pi(s) = \mathbb{E}_\pi \left[ r_t + \gamma G_{t+1} \mid S_t = s \right]$$
Step 1: Expand over policy action choices:
$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \mathbb{E}_\pi \left[ r_t + \gamma G_{t+1} \mid S_t = s, A_t = a \right]$$
Step 2: Expand over transition dynamics to next states $s'$ and immediate rewards $R(s, a)$:
$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[ R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \mathbb{E}_\pi \left[ G_{t+1} \mid S_{t+1} = s' \right] \right]$$
Step 3: Substitute definition of $V^\pi(s')$:
$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[ R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^\pi(s') \right]$$

### Bellman Expectation Equation for $Q^\pi(s, a)$
Decomposing action-value into immediate reward and discounted next state-value:
$$Q^\pi(s, a) = R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^\pi(s')$$
Substituting the bridge relationship $V^\pi(s') = \sum_{a'} \pi(a' \mid s') Q^\pi(s', a')$:
$$Q^\pi(s, a) = R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \sum_{a' \in \mathcal{A}} \pi(a' \mid s') Q^\pi(s', a')$$

### Two-Step Backup Decomposition
The Bellman expectation backup traverses two successive probability layers:
1. **Action Selection Layer**: $\sum_a \pi(a \mid s)$ accounts for policy stochasticity.
2. **Environmental Transition Layer**: $\sum_{s'} P(s' \mid s, a)$ accounts for dynamic environment stochasticity.

### Exact Matrix Formulation & Closed-Form Analytic Solution
For finite discrete MDPs with $|\mathcal{S}| = N$, define:
- State-value column vector: $\boldsymbol{v}_\pi = [V^\pi(s_1), \ldots, V^\pi(s_N)]^T \in \mathbb{R}^N$
- Policy-averaged reward vector: $\boldsymbol{r}_\pi = [r_\pi(s_1), \ldots, r_\pi(s_N)]^T \in \mathbb{R}^N$ where $r_\pi(s) = \sum_a \pi(a \mid s) R(s, a)$
- Policy-averaged transition probability matrix: $\boldsymbol{P}_\pi \in \mathbb{R}^{N \times N}$ where $(\boldsymbol{P}_\pi)_{ij} = \sum_a \pi(a \mid s_i) P(s_j \mid s_i, a)$

The Bellman expectation equation becomes:
$$\boldsymbol{v}_\pi = \boldsymbol{r}_\pi + \gamma \boldsymbol{P}_\pi \boldsymbol{v}_\pi$$
Rearranging:
$$(\boldsymbol{I} - \gamma \boldsymbol{P}_\pi) \boldsymbol{v}_\pi = \boldsymbol{r}_\pi$$
Since $\boldsymbol{P}_\pi$ is a row-stochastic transition matrix ($\sum_j (\boldsymbol{P}_\pi)_{ij} = 1$) with spectral radius $\rho(\boldsymbol{P}_\pi) \le 1$, and $\gamma \in [0, 1)$, the matrix $(\boldsymbol{I} - \gamma \boldsymbol{P}_\pi)$ is strictly diagonally dominant and guaranteed to be invertible. The closed-form analytical solution is:
$$\boldsymbol{v}_\pi = (\boldsymbol{I} - \gamma \boldsymbol{P}_\pi)^{-1} \boldsymbol{r}_\pi$$
*Computational Complexity*: Inverting an $N \times N$ matrix costs $\mathcal{O}(N^3)$, which is intractable for large state spaces ($N > 10^4$), necessitating iterative methods (Dynamic Programming, TD learning).

---

## 2.7 Bellman Optimality Equations & Fixed-Point Theory

An optimal policy $\pi^*$ achieves state values greater than or equal to all policies: $V^*(s) = \max_\pi V^\pi(s)$ for all $s \in \mathcal{S}$.

### Bellman Optimality Equations
$$V^*(s) = \max_{a \in \mathcal{A}} Q^*(s, a) = \max_{a \in \mathcal{A}} \left[ R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^*(s') \right]$$
$$Q^*(s, a) = R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \max_{a' \in \mathcal{A}} Q^*(s', a')$$

### Optimal Policy Extraction
Once $Q^*(s, a)$ is determined, the optimal deterministic policy is directly extracted via greedy selection:
$$\pi^*(s) = \arg\max_{a \in \mathcal{A}} Q^*(s, a)$$

### Bellman Operator & Banach Fixed Point Contraction Proof
Define the Bellman optimality operator $T^*: \mathbb{R}^N \to \mathbb{R}^N$ acting on arbitrary value function $V$:
$$(T^* V)(s) = \max_{a \in \mathcal{A}} \left[ R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V(s') \right]$$

#### Theorem: $T^*$ is a $\gamma$-contraction mapping under the $L_\infty$ norm
$$\lVert T^* u - T^* v \rVert_\infty \le \gamma \lVert u - v \rVert_\infty, \quad \forall u, v \in \mathbb{R}^N$$

#### Proof
For any state $s$:
$$|(T^* u)(s) - (T^* v)(s)| = \left| \max_a \left( R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) u(s') \right) - \max_{a'} \left( R(s, a') + \gamma \sum_{s'} P(s' \mid s, a') v(s') \right) \right|$$
Using the algebraic inequality $|\max_x f(x) - \max_x g(x)| \le \max_x |f(x) - g(x)|$:
$$\le \max_a \left| \left( R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) u(s') \right) - \left( R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) v(s') \right) \right|$$
$$= \max_a \gamma \left| \sum_{s'} P(s' \mid s, a) (u(s') - v(s')) \right|$$
$$\le \gamma \max_a \sum_{s'} P(s' \mid s, a) |u(s') - v(s')| \le \gamma \max_a \sum_{s'} P(s' \mid s, a) \lVert u - v \rVert_\infty$$
Since transition probabilities sum to 1 ($\sum_{s'} P(s' \mid s, a) = 1$):
$$|(T^* u)(s) - (T^* v)(s)| \le \gamma \lVert u - v \rVert_\infty$$
Taking the supremum over all $s \in \mathcal{S}$:
$$\lVert T^* u - T^* v \rVert_\infty \le \gamma \lVert u - v \rVert_\infty \quad \blacksquare$$

#### Corollaries of Banach Fixed Point Theorem
1. **Existence and Uniqueness**: $T^* V^* = V^*$ has a unique fixed point $V^*$.
2. **Convergence of Value Iteration**: Starting from any arbitrary initial vector $V_0$, the sequence $V_{k+1} = T^* V_k$ converges geometrically to $V^*$ at rate $\gamma$:
   $$\lVert V_k - V^* \rVert_\infty \le \frac{\gamma^k}{1 - \gamma} \lVert V_1 - V_0 \rVert_\infty$$

---

## 2.8 Dynamic Programming: Policy Iteration & Value Iteration

Dynamic programming solves known MDPs ($P$ and $R$ known) by turning Bellman equations into update rules.

### 1. Policy Iteration
Alternates two decoupled steps until policy stabilization:
1. **Policy Evaluation**: Compute $V^{\pi_k}$ by repeatedly sweeping the Bellman expectation operator:
   $$V_{m+1}(s) \leftarrow \sum_a \pi_k(a \mid s) \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V_m(s') \right]$$
2. **Policy Improvement**: Update policy greedily with respect to evaluated values:
   $$\pi_{k+1}(s) \leftarrow \arg\max_{a \in \mathcal{A}} \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V^{\pi_k}(s') \right]$$
*Convergence*: Terminates in a finite number of iterations because the number of deterministic policies $|\mathcal{A}|^{|\mathcal{S}|}$ is finite.

### 2. Value Iteration
Combines evaluation and improvement into a single step by directly applying the Bellman optimality operator:
$$V_{k+1}(s) \leftarrow \max_{a \in \mathcal{A}} \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V_k(s') \right]$$
*Stopping Criterion*: Terminate when $\lVert V_{k+1} - V_k \rVert_\infty < \theta \frac{1 - \gamma}{2\gamma}$, ensuring $\lVert V_{k+1} - V^* \rVert_\infty < \theta$.

---

## 2.9 Q-Learning: Off-Policy TD Control & Exploratory Policies

When environment dynamics $P(s' \mid s, a)$ are unknown (black box), the agent must learn from interaction samples $(s_t, a_t, r_t, s_{t+1})$.

### The Tabular Q-Learning Update Rule
$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ \underbrace{r_t + \gamma \max_{a'} Q(s_{t+1}, a')}_{\text{TD Target}} - Q(s_t, a_t) \right]$$
Where the Temporal Difference Error is defined as:
$$\delta_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)$$

### Off-Policy Mechanism
- **Behavior Policy ($\mu$)**: Generates actions during interaction to guarantee continuous exploration (e.g., $\epsilon$-greedy):
  $$\mu(a \mid s) = \begin{cases} 1 - \epsilon + \frac{\epsilon}{|\mathcal{A}|} & \text{if } a = \arg\max_a Q(s, a) \\ \frac{\epsilon}{|\mathcal{A}|} & \text{otherwise} \end{cases}$$
- **Target Policy ($\pi$)**: Evaluated and improved by the max operator; strictly greedy ($\pi(s) = \arg\max_a Q(s, a)$).
- **Core Benefit**: Q-learning learns the optimal policy $Q^*$ regardless of the exploratory choices made by the behavior policy.

### The CliffWalking Benchmark: Q-Learning vs. SARSA
In CliffWalking, an agent navigates a grid from Start to Goal. A cliff spans the bottom edge between Start and Goal (stepping into the cliff incurs reward $-100$ and teleports to Start; normal steps incur reward $-1$):
- **Q-Learning (Off-Policy)**: Optimizes the greedy target policy, finding the shortest path directly along the edge of the cliff. However, during training with $\epsilon$-greedy exploration, random exploratory actions frequently cause the agent to fall off the cliff, depressing training return.
- **SARSA (On-Policy)**: Update rule uses the action actually selected by the behavior policy:
  $$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]$$
  SARSA learns the *safe path* around the top of the grid, intentionally avoiding the cliff edge to mitigate the risk of exploratory missteps.

---

## 2.10 The Triad: Dynamic Programming vs. Monte Carlo vs. Temporal Difference

Reinforcement learning value estimation spans a fundamental structural triangle defined by sampling (experience vs. model) and bootstrapping (one-step recursion vs. full rollout):

```
                       Full Expectation (No Sampling)
                                     |
                                     |
                           Dynamic Programming (DP)
                                     |
             Bootstrapping ----------+---------- Full Rollout (No Bootstrap)
                   |                                   |
                   |                                   |
         Temporal Difference (TD)                Monte Carlo (MC)
                   |                                   |
                   +-----------------------------------+
                                     |
                        Sampled Experience (Model-Free)
```

### 1. Dynamic Programming (DP)
- **Target**: $V(s) \leftarrow \sum_a \pi(a \mid s) [R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V(s')]$
- **Mechanics**: Model-based, full expectation across all branches, bootstraps on existing value estimates. Zero sampling variance, requires complete environment model $P$ and $R$.

### 2. Monte Carlo (MC)
- **Target**: $V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]$ where $G_t = \sum_{k=0}^T \gamma^k r_{t+k}$
- **Mechanics**: Model-free, samples complete trajectory episodes, updates using actual return $G_t$.
- **Tradeoff**: **Zero Bias** (empirical return is an unbiased estimator of true expected return), but **High Variance** (accumulates stochasticity of all actions and state transitions across the full episode). Requires episodic tasks.

### 3. Temporal Difference (TD(0))
- **Target**: $V(S_t) \leftarrow V(S_t) + \alpha [r_t + \gamma V(S_{t+1}) - V(S_t)]$
- **Mechanics**: Model-free, samples single transition $(s, a, r, s')$, bootstraps immediately using estimated next-state value $V(S_{t+1})$.
- **Tradeoff**: **Low Variance** (depends on one-step randomness), but **Biased** (depends on imperfect initial value estimates $V(S_{t+1})$). Can learn online in continuing, non-terminating tasks.

### Systematic Comparison Matrix

| Evaluation Dimension | Dynamic Programming (DP) | Monte Carlo (MC) | Temporal Difference (TD(0)) |
| :--- | :--- | :--- | :--- |
| **Model Requirements** | Requires known $P(s' \mid s, a)$ & $R(s, a)$ | Fully Model-Free | Fully Model-Free |
| **Bootstrapping** | Yes (uses successor state values) | No (uses realized empirical return $G_t$) | Yes (uses successor state value $V(s')$) |
| **Sampling** | No (sweeps full transition tree) | Yes (samples complete episode rollouts) | Yes (samples single transitions) |
| **Bias** | Zero (exact expectation under model) | **Zero Bias** ($\mathbb{E}[G_t \mid s] = V^\pi(s)$) | **Biased** (depends on error in $V(s')$) |
| **Variance** | Zero | **High Variance** (compounds across $T$) | **Low Variance** (single step noise only) |
| **Update Frequency** | After exhaustive state sweep | Only at episode termination | **Step-by-step online** |
| **Applicable Tasks** | Finite known models | Episodic tasks only | Episodic and continuing tasks |
| **Convergence** | $\lim_{k \to \infty} V_k = V^*$ (contraction) | Asymptotic via Law of Large Numbers | Asymptotic under Robbins-Monro conditions |

---

## 2.11 Reinforcement Learning Taxonomy & Data Regimes

Reinforcement learning algorithms are structured across two orthogonal axes:
1. **Policy Relationship (On-Policy vs. Off-Policy)**: Who generates the data vs. who learns?
2. **Data Availability (Online vs. Offline)**: Can the agent interact with the environment in real-time, or is it restricted to a static dataset?

```
+------------------------------------+------------------------------------+
|                                    |                                    |
|       ONLINE ON-POLICY             |         ONLINE OFF-POLICY          |
|  - Actor interacts and learns      |  - Explores via behavior policy    |
|    from its own fresh data         |  - Reuses past replay data         |
|  - High sample cost, high stability|  - High sample efficiency          |
|  * Algorithms: SARSA, A2C, PPO     |  * Algorithms: Q-Learning, DQN, SAC|
|                                    |                                    |
+------------------------------------+------------------------------------+
|                                    |                                    |
|       OFFLINE ON-POLICY            |         OFFLINE OFF-POLICY         |
|  - Theoretical edge case:          |  - Fixed static dataset D          |
|    Fixed dataset collected         |  - Severe distribution shift       |
|    by the exact target policy      |  - Out-of-distribution actions     |
|  * Rarely applicable in practice   |  * Algorithms: CQL, IQL, BCQ       |
|                                    |                                    |
+------------------------------------+------------------------------------+
```

### The LLM RL Training-Inference Mismatch
In modern Large Language Model post-training (e.g., PPO or GRPO for LLMs), rollouts are generated asynchronously on inference engine clusters (vLLM, TensorRT-LLM) using policy snapshot $\pi_{\theta_{\text{rollout}}}$. Meanwhile, training occurs on distributed clusters (Megatron-LM, FSDP) updating parameters $\theta$. By the time the gradient update occurs, $\theta \neq \theta_{\text{rollout}}$. This produces an inherent **Off-Policy Gap**, requiring importance sampling corrections:
$$w_t = \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\theta_{\text{rollout}}}(y_t \mid x, y_{<t})}$$

---

## 2.12 Reward Design, Potential Shaping, & Modern Alignment Models

The reward function $R(s, a, s')$ defines the optimization objective. Misaligned reward signals lead to catastrophic agent failure modes.

### Potential-Based Reward Shaping
Reward shaping augments a sparse environment reward $R(s, a, s')$ with an auxiliary shaping term $F(s, a, s')$:
$$R'(s, a, s') = R(s, a, s') + F(s, a, s')$$

#### Theorem (Ng, Harada, & Russell, 1999)
A reward shaping function $F(s, a, s')$ preserves the policy ordering (ensuring that every optimal policy $\pi^*$ of the shaped MDP is also an optimal policy of the original MDP) if and only if $F$ is expressible as the discounted difference of a potential function $\Phi: \mathcal{S} \to \mathbb{R}$:
$$F(s, a, s') = \gamma \Phi(s') - \Phi(s)$$

#### Proof
Consider the return along any trajectory under $R'$:
$$G_t' = \sum_{k=0}^{\infty} \gamma^k R'(s_{t+k}, a_{t+k}, s_{t+k+1}) = \sum_{k=0}^{\infty} \gamma^k \left[ R(s_{t+k}, a_{t+k}, s_{t+k+1}) + \gamma \Phi(s_{t+k+1}) - \Phi(s_{t+k}) \right]$$
Expanding the telescoping sum:
$$G_t' = \sum_{k=0}^{\infty} \gamma^k R(s_{t+k}, a_{t+k}, s_{t+k+1}) + \sum_{k=0}^{\infty} \gamma^{k+1} \Phi(s_{t+k+1}) - \sum_{k=0}^{\infty} \gamma^k \Phi(s_{t+k})$$
$$= G_t + \left( \gamma \Phi(s_{t+1}) + \gamma^2 \Phi(s_{t+2}) + \cdots \right) - \left( \Phi(s_t) + \gamma \Phi(s_{t+1}) + \gamma^2 \Phi(s_{t+2}) + \cdots \right)$$
All intermediate potential terms cancel exactly:
$$G_t' = G_t - \Phi(s_t)$$
Because $\Phi(s_t)$ depends strictly on the initial state $s_t$ and is independent of the actions chosen by the policy, maximizing $\mathbb{E}[G_t']$ is strictly mathematically equivalent to maximizing $\mathbb{E}[G_t]$. The optimal policy is invariant to potential-based shaping: $\pi^{*\prime} \equiv \pi^*$. $\blacksquare$

### Reward Misspecification & Reward Hacking Failures
When rewards do not follow potential-based formulation, agents exploit unintended shortcuts:
1. **CoastRunners Boat Racing**: Agent discovered that spinning in circles to continuously knock over target buoys accumulated higher scores than finishing the race, repeatedly catching fire and crashing rather than crossing the finish line.
2. **Tetris Pause Exploit**: An agent facing inevitable block stacking overflow paused the game indefinitely to prevent the terminal $-1$ penalty.
3. **Robotic Grasping**: A robotic arm trained to grasp an object by minimizing the distance between gripper and block learned to place the gripper directly between the block and the camera, occluding the view and tricking the visual reward detector.

### Modern Reward Modeling in Large-Scale Alignment
1. **Bradley-Terry Preference Model**:
   Given prompt $x$ and pair of candidate responses $(y_w, y_l)$ where human or AI judges prefer $y_w \succ y_l$, the probability of preference is modeled via scalar reward function $r_\psi$:
   $$P(y_w \succ y_l \mid x) = \sigma(r_\psi(x, y_w) - r_\psi(x, y_l)) = \frac{1}{1 + e^{-(r_\psi(x, y_w) - r_\psi(x, y_l))}}$$
   Trained via negative log-likelihood loss:
   $$\mathcal{L}_{\text{RM}}(\psi) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma(r_\psi(x, y_w) - r_\psi(x, y_l)) \right]$$
2. **Process Reward Models (PRM) vs. Outcome Reward Models (ORM)**:
   - *ORM*: Evaluates only the final response $r(x, y)$, providing coarse, delayed feedback prone to reward hacking (getting the right answer through flawed reasoning).
   - *PRM*: Assigns rewards to every individual step of a multi-step chain-of-thought: $r(x, y_1, \ldots, y_t)$, enabling precise step-level credit assignment.
3. **Group Relative Policy Optimization (GRPO)**:
   Eliminates the separate critic network entirely. For each prompt $q$, the policy generates a group of $G$ candidate outputs $\{o_1, o_2, \ldots, o_G\}$. The advantage of candidate $o_i$ is computed by standardizing rewards across the group:
   $$\hat{A}_i = \frac{r(q, o_i) - \text{mean}(\{r(q, o_1), \ldots, r(q, o_G)\})}{\text{std}(\{r(q, o_1), \ldots, r(q, o_G)\}) + \epsilon}$$
   GRPO drastically reduces memory footprint during distributed RL fine-tuning.


# 3. Chapter 07: Deep Q-Networks (DQN)

### Sources
- `docs/en/chapter07_dqn/from-q-to-dqn.md`
- `docs/en/chapter07_dqn/dqn-components.md`
- `docs/en/chapter07_dqn/lunar-lander.md`
- `docs/en/chapter07_dqn/dqn-family.md`
- `docs/en/chapter07_dqn/visual-game-projects.md`

---

## 3.1 Transition from Tabular Q-Learning to Function Approximation

Tabular Q-learning stores action-values in a discrete matrix of dimension $|\mathcal{S}| \times |\mathcal{A}|$. In realistic physical systems and visual gaming environments, tabular representations break down entirely:
1. **Curse of Dimensionality**: Continuous state spaces (e.g., CartPole with $\mathcal{S} \subseteq \mathbb{R}^4$, LunarLander with $\mathcal{S} \subseteq \mathbb{R}^8$) or pixel observation spaces (Atari screen with $210 \times 160 \times 3$ RGB pixels yielding $256^{100,800}$ distinct states) cannot be enumerated in memory.
2. **Absence of Generalization**: In a Q-table, visiting state $s$ provides zero updates to neighboring state $s + \Delta s$. Real control requires generalizing value estimates across geometrically and semantically similar configurations.

### Deep Function Approximation
A parameterized neural network $Q(s, a; \theta)$ with weight parameters $\theta$ approximates the optimal action-value function:
$$Q(s, a; \theta) \approx Q^*(s, a)$$
Given high-dimensional state input $s$, the network outputs a $|\mathcal{A}|$-dimensional vector containing predicted Q-values for all discrete actions simultaneously in a single forward pass.

---

## 3.2 Instability Phenomena: Moving Targets & Correlated Transitions

When tabular Q-learning is naively converted to neural network parameterization by applying online gradient descent on single transitions $(s_t, a_t, r_t, s_{t+1})$:
$$\theta \leftarrow \theta - \alpha \left( r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta) - Q(s_t, a_t; \theta) \right) \nabla_\theta Q(s_t, a_t; \theta)$$
Training almost universally diverges due to two fundamental structural pathologies:

### 1. Strongly Correlated Transitions (Violation of i.i.d.)
Standard deep learning optimization (Stochastic Gradient Descent) strictly assumes training data are independent and identically distributed (i.i.d.). In an agent trajectory, consecutive states $s_t$ and $s_{t+1}$ are strongly auto-correlated. Updating network weights on successive trajectory steps causes catastrophic interference: weights adjust to optimize the local neighborhood while catastrophically forgetting previously learned regions of the state space.

### 2. Moving Targets (Non-Stationary Objective)
In supervised deep learning, targets $y$ are fixed labels. In Q-learning, the regression target is:
$$y_t = r_t + \gamma \max_{a'} Q(s_{t+1}, a'; \theta)$$
Crucially, $y_t$ is a direct function of the very parameters $\theta$ being updated. As the optimizer updates $\theta$ to pull $Q(s_t, a_t; \theta)$ toward $y_t$, the target $y_t$ simultaneously shifts. This "dog chasing its own tail" dynamic produces uncontrolled positive feedback loops, runaway gradient explosion, or oscillation.

---

## 3.3 The Two Pillars of DQN: Experience Replay & Target Networks

Deep Q-Networks (Mnih et al., Nature 2015) solved these instability pathologies via two architectural innovations:

### Pillar 1: Experience Replay Buffer ($\mathcal{D}$)
Transitions are collected and appended to a large cyclic FIFO buffer of capacity $N$ (e.g., $N = 10^5$ to $10^6$ transitions):
$$e_t = (s_t, a_t, r_t, s_{t+1}, d_t) \in \mathcal{D}$$
Where $d_t \in \{0, 1\}$ is the episode termination flag (`done`). During network updates, mini-batches of size $B$ (e.g., $B = 32$ or $64$) are sampled uniformly at random:
$$\mathcal{B} \sim \mathcal{U}(\mathcal{D})$$
- **Decorrelation**: Random sampling breaks the temporal sequence of transitions, reconstructing an empirical i.i.d. distribution.
- **Data Efficiency**: Rare transitions (e.g., successful landings or high-reward events) are reused across multiple gradient updates rather than discarded after a single step.

### Pillar 2: Target Network ($Q(\cdot, \cdot; \theta^-)$)
A duplicate neural network with parameters $\theta^-$ is maintained exclusively to compute regression targets:
$$y = r + \gamma (1 - d) \max_{a'} Q(s', a'; \theta^-)$$
The target network weights $\theta^-$ are held completely stationary during standard gradient updates. Parameter synchronization follows one of two schemes:
1. **Periodic Hard Synchronization**:
   $$\theta^- \leftarrow \theta \quad \text{every } C \text{ environment steps (e.g., } C = 1,000 \text{ steps)}$$
2. **Polyak Soft Update (Exponential Moving Average)**:
   $$\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^- \quad \text{every training step with } \tau \ll 1 \text{ (e.g., } \tau = 0.005\text{)}$$
Freezing $\theta^-$ stabilizes the optimization manifold, converting the moving target problem into a series of quasi-stationary regression targets.

---

## 3.4 Loss Formulations: MSE vs. Huber / Smooth L1

Given transition batch $(s_i, a_i, r_i, s'_i, d_i)$, define the TD error:
$$\delta_i = \left( r_i + \gamma (1 - d_i) \max_{a'} Q(s'_i, a'; \theta^-) \right) - Q(s_i, a_i; \theta)$$

### 1. Mean Squared Error (MSE) Loss
$$\mathcal{L}_{\text{MSE}}(\theta) = \frac{1}{|B|} \sum_{i=1}^{|B|} \delta_i^2$$
Gradient: $\nabla_\theta \mathcal{L}_{\text{MSE}} = - \frac{2}{|B|} \sum_i \delta_i \nabla_\theta Q(s_i, a_i; \theta)$.
*Weakness*: When $\delta_i$ is large (e.g., sudden collision penalties or reward spikes), the quadratic term causes massive gradient norms that destabilize deep networks.

### 2. Huber Loss (Smooth L1 Loss)
$$\mathcal{L}_{\text{Huber}}(\delta) = \begin{cases} \frac{1}{2} \delta^2 & \text{if } |\delta| \le 1.0 \\ |\delta| - \frac{1}{2} & \text{otherwise} \end{cases}$$
The derivative with respect to $\delta$ is:
$$\frac{\partial \mathcal{L}_{\text{Huber}}}{\partial \delta} = \begin{cases} \delta & \text{if } |\delta| \le 1.0 \\ \text{sign}(\delta) & \text{otherwise} \end{cases} = \text{clip}(\delta, -1.0, 1.0)$$
*Strength*: Huber loss behaves quadratically for small errors (smooth convergence near zero) and linearly for large errors (bounded gradient magnitude $\le 1$), providing robustness against outlier transitions.

---

## 3.5 Complete Deep Q-Network Algorithm & Control Flow

The formal algorithmic sequence executes as follows:

```
Algorithm: Deep Q-Network (DQN) with Replay Buffer and Target Network
========================================================================================
1: Initialize replay buffer D to capacity N
2: Initialize action-value network Q with random weights theta
3: Initialize target network Q^- with weights theta^- = theta
4: for episode = 1 to M do
5:     Initialize environment, observe s_1
6:     for t = 1 to T do
7:         Select action a_t via epsilon-greedy:
8:             With probability epsilon select random action a_t in A
9:             Otherwise select a_t = argmax_a Q(s_t, a; theta)
10:        Execute action a_t, observe reward r_t, next state s_{t+1}, and done d_t
11:        Store transition (s_t, a_t, r_t, s_{t+1}, d_t) into buffer D
12:        Sample random mini-batch of B transitions (s_i, a_i, r_i, s'_i, d_i) from D
13:        Compute target:
14:            y_i = r_i if d_i == 1 else r_i + gamma * max_{a'} Q(s'_i, a'; theta^-)
15:        Compute Huber loss:
16:            L(theta) = (1/B) * sum_i HuberLoss(y_i - Q(s_i, a_i; theta))
17:        Perform gradient descent step on theta with optimizer Adam
18:        Every C steps, update target network: theta^- <- theta
19:        s_t <- s_{t+1}
20:        if d_t is True then break
21:    end for
22:    Decay exploration rate: epsilon <- max(epsilon_end, epsilon * epsilon_decay)
23: end for
========================================================================================
```

---

## 3.6 The DQN Improvement Family (Rainbow Innovations)

Over several years, six foundational modifications were proposed to resolve specific theoretical weaknesses of the original DQN, eventually synthesized into **Rainbow**:

```
                       +---------------------------------------+
                       |              DQN (2015)               |
                       +---------------------------------------+
                                          |
          +-------------------------------+-------------------------------+
          |               |               |               |               |
          v               v               v               v               v
    Double DQN       Dueling DQN        PER          Multi-Step      Distributional
   (Decouples Q)    (V & A Split)  (Error-Weighted)   (N-Step)           (C51)
          |               |               |               |               |
          +---------------+---------------+---------------+---------------+
                                          |
                                          v
                               +---------------------+
                               |   Rainbow (2018)    |
                               +---------------------+
```

### 1. Double DQN (Decoupling Action Selection from Evaluation)
- **Problem**: Standard DQN computes targets using $\max_{a'} Q(s', a'; \theta^-)$. Because of stochastic noise and estimation errors, the maximum of noisy estimates systematically overestimates true action values:
  $$\mathbb{E}\left[ \max_a \hat{Q}(s, a) \right] \ge \max_a \mathbb{E}\left[ \hat{Q}(s, a) \right]$$
  This overestimation bias compounds recursively across Bellman updates.
- **Solution**: Decouple the action selection from the value evaluation. Use the online network $\theta$ to select the best action, and use the target network $\theta^-$ to evaluate its value:
  $$a^* = \arg\max_{a'} Q(s', a'; \theta)$$
  $$y_{\text{Double}} = r + \gamma (1 - d) Q\left( s', a^*; \theta^- \right) = r + \gamma (1 - d) Q\left( s', \arg\max_{a'} Q(s', a'; \theta); \theta^- \right)$$

### 2. Dueling DQN (State Value & Advantage Stream Decomposition)
- **Problem**: In many states, the absolute value of the state $V(s)$ is paramount, while the relative differences between actions $A(s, a)$ are minor (e.g., whether a car moves forward or slightly turns when no obstacles exist). Standard DQN learns full Q-values for every action separately, requiring redundant training across all actions.
- **Solution**: Decompose the final hidden layers into two parallel streams:
  1. Scalar state-value stream: $V(s; \theta, \beta) \in \mathbb{R}^1$
  2. Action advantage stream: $A(s, a; \theta, \alpha) \in \mathbb{R}^{|\mathcal{A}|}$
- **Identifiability Formulation**: A naive sum $Q(s, a) = V(s) + A(s, a)$ is unidentifiable because adding any constant $c$ to $V$ and subtracting $c$ from $A$ gives identical $Q$. To enforce uniqueness and stability, Dueling DQN subtracts the mean advantage:
  $$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a'; \theta, \alpha) \right)$$
  Subtracting the mean ensures the advantage stream has zero mean across actions while allowing backpropagation to directly shift state values across all action channels simultaneously.

### 3. Prioritized Experience Replay (PER)
- **Problem**: Uniform random sampling gives equal probability to all transitions in $\mathcal{D}$, wasting compute on transitions that the network has already mastered.
- **Solution**: Sample transitions with probability proportional to their TD error magnitude $|\delta_i|$.
  - Priority assignment:
    $$p_i = |\delta_i| + \epsilon_p$$
    Where $\epsilon_p > 0$ (e.g., $10^{-5}$) ensures transitions with zero error retain non-zero sampling probability.
  - Sampling distribution:
    $$P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \quad \alpha \in [0, 1]$$
    When $\alpha = 0$, this reduces to uniform sampling; when $\alpha = 1$, it is fully proportional to priority.
  - **Bias Correction via Importance Sampling**: Because prioritized sampling changes the state visitation distribution, it introduces estimation bias. PER corrects this using importance sampling weights:
    $$w_i = \left( \frac{1}{N \cdot P(i)} \right)^\beta, \quad \text{normalized by } \max_j w_j$$
    Hyperparameter $\beta \in [0, 1]$ is annealed linearly from $\beta_0 = 0.4$ to $1.0$ at the end of training.
  - Weighted loss update:
    $$\mathcal{L}_{\text{PER}}(\theta) = \frac{1}{|B|} \sum_{i \in B} w_i \mathcal{L}_{\text{Huber}}(\delta_i)$$

### 4. Multi-Step Returns ($N$-Step Q-Learning)
Replaces the 1-step Bellman target with an $N$-step accumulated reward target:
$$y_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n \max_{a'} Q(s_{t+n}, a'; \theta^-)$$
- **Tradeoff**: Multi-step returns propagate reward signals backward $N$ times faster, reducing bias from the bootstrapping critic. However, for large $N$, off-policy validity breaks down unless importance sampling corrections are applied. In practice, $N = 3$ provides the optimal empirical balance.

### 5. Distributional Reinforcement Learning (C51)
Instead of predicting the scalar expectation $Q(s, a) = \mathbb{E}[Z(s, a)]$, C51 models the full probability distribution of returns $Z(s, a)$ over a fixed support of $N_{\text{atoms}} = 51$ discrete bins $\{z_0, z_1, \ldots, z_{50}\}$ uniformly spaced in $[V_{\min}, V_{\max}]$:
$$Z(s, a) \overset{D}{=} R(s, a) + \gamma Z(S', A^*)$$
The network outputs softmax probabilities $p_i(s, a) = \mathbb{P}(Z(s, a) = z_i)$. Targets are shifted by the Bellman operator and projected back onto the discrete support, optimizing the Cross-Entropy loss. This preserves multimodality and variance information that scalar expectations destroy.

### 6. Noisy Networks for Exploration (NoisyNet)
Replaces the heuristic $\epsilon$-greedy exploration policy by introducing learnable zero-mean Gaussian noise directly into the linear layer parameters:
$$y = (W + \sigma_W \odot \epsilon_W) x + (b + \sigma_b \odot \epsilon_b)$$
Where $\mu_W, \mu_b$ are learned mean weights, $\sigma_W, \sigma_b$ are learned noise scaling parameters, and $\epsilon_W, \epsilon_b$ are sampled stochastic variables. The agent learns state-dependent, temporally extended exploration strategies that adapt automatically throughout training.

### 7. Intrinsic Curiosity & Non-Task Exploration
When extrinsic rewards $r_t^{\text{extrinsic}}$ are extremely sparse:
$$r_t^{\text{total}} = r_t^{\text{extrinsic}} + \beta r_t^{\text{intrinsic}}$$
- **Intrinsic Curiosity Module (ICM)**: Encodes states into feature representations $\phi(s)$ via an inverse dynamics model, then trains a forward dynamics model $\hat{\phi}(s_{t+1}) = f(\phi(s_t), a_t)$. The prediction error serves as intrinsic reward:
  $$r_t^{\text{intrinsic}} = \frac{1}{2} \lVert \phi(s_{t+1}) - \hat{\phi}(s_{t+1}) \rVert_2^2$$
- **Random Network Distillation (RND)**: Utilizes a fixed, randomly initialized target neural network $f^*: \mathcal{S} \to \mathbb{R}^k$ and trains a predictor network $f_\theta: \mathcal{S} \to \mathbb{R}^k$. The intrinsic reward is the distillation prediction error:
  $$r_t^{\text{intrinsic}} = \lVert f_\theta(s_t) - f^*(s_t) \rVert_2^2$$
  Unvisited novel states generate high prediction error, incentivizing the agent to explore unseen configurations without relying on environmental task rewards.

---

## 3.7 Continuous Observation Benchmark: LunarLander-v2

LunarLander-v2 serves as the primary continuous-observation, discrete-action control benchmark for deep Q-learning.

### Environment Specification
- **Observation Space**: $\mathcal{S} \subseteq \mathbb{R}^8$ continuous vector:
  1. $x$: horizontal coordinate ($0.0 = \text{center}$)
  2. $y$: vertical coordinate ($0.0 = \text{ground level}$)
  3. $v_x$: horizontal velocity
  4. $v_y$: vertical velocity
  5. $\theta$: lander body angle ($0.0 = \text{vertical}$)
  6. $\omega$: angular velocity
  7. $l_1 \in \{0, 1\}$: left leg ground contact boolean flag
  8. $l_2 \in \{0, 1\}$: right leg ground contact boolean flag
- **Action Space**: $\mathcal{A} = \{0, 1, 2, 3\}$ discrete thruster firings:
  - `0`: Do nothing (free fall under lunar gravity)
  - `1`: Fire left orientation thruster (rotates lander right)
  - `2`: Fire main thruster (provides upward vertical thrust)
  - `3`: Fire right orientation thruster (rotates lander left)
- **Reward Dynamics**:
  - Landing between flags: $+100$ to $+140$ points based on proximity to origin $(0, 0)$.
  - Leg ground contact: $+10$ points per leg.
  - Crashing / Out-of-bounds: $-100$ terminal penalty.
  - Fuel consumption penalty: Main thruster costs $-0.30$ per frame; side thrusters cost $-0.03$ per frame.
  - **Solved Criterion**: Average return $\ge 200.0$ across 100 consecutive evaluation episodes.

### Failure Modes & Debugging Triage
1. **Engine Hovering Loop**: Agent learns that firing the main engine prevents crashing (avoiding the $-100$ penalty) but never descends to touch the pad, hovering until the 1,000-step truncation timeout is reached.
   - *Fix*: Ensure step penalties exist or increase $\gamma$ to discount distant rewards.
2. **Rotational Instability**: Angular velocity $\omega$ diverges, causing the lander to spin uncontrollably out of bounds.
   - *Fix*: Check observation normalization; verify that replay buffer capacity is large enough to retain recovering transitions.
3. **Premature Convergence to Crash**: Agent crashes quickly before replay buffer accumulates successful landing demonstrations.
   - *Fix*: Slow down $\epsilon$ decay schedule to preserve exploration over at least $50,000$ steps.

---

## 3.8 Visual Game Projects: Atari Preprocessing, Architecture, & Task Frontiers

When learning directly from raw visual pixels (e.g., Atari 2600 games via Arcade Learning Environment), the standard DQN architecture and preprocessing pipeline are strictly standardized:

```
[Raw Frame 210x160x3 RGB]
         |
         v
[Grayscale Conversion & Bilinear Resize -> 84x84 Grayscale]
         |
         v
[Frame Skip (k=4): Repeat action for 4 frames, take pixel-wise max over last 2]
         |
         v
[Frame Stacking: Concatenate last 4 processed frames -> Tensor (4, 84, 84)]
         |
         v
[Nature DQN CNN Trunk]
  |-- Conv2D(32 filters, 8x8 kernel, stride 4) + ReLU
  |-- Conv2D(64 filters, 4x4 kernel, stride 2) + ReLU
  |-- Conv2D(64 filters, 3x3 kernel, stride 1) + ReLU
  |-- Flatten -> Linear(512 units) + ReLU
  +-- Linear(512, |A| actions) -> Q(s, a)
```

### Atari Preprocessing Pipeline Rationale
1. **Grayscale & Downsampling**: Eliminates redundant color channels ($210 \times 160 \times 3 \to 84 \times 84$), reducing input dimensionality by over $93\%$ while preserving spatial geometry.
2. **Frame Skipping ($k=4$)**: Human reflexes do not operate at 60 Hz. Computing agent decisions every 4 frames reduces compute load by $4\times$ while expanding the effective planning horizon.
3. **Max-Pooling Over Consecutive Frames**: Many Atari 2600 games exploit phosphor persistence by rendering sprites on alternate frames to simulate transparency (e.g., bullets in Space Invaders). Taking the pixel-wise maximum over frames $t$ and $t-1$ eliminates sprite flickering artifacts.
4. **Frame Stacking ($m=4$)**: A single static 2D image cannot reveal velocity, acceleration, or trajectory direction (e.g., is the Pong ball moving toward the paddle or away?). Stacking 4 successive frames constructs an artificial state representation that satisfies the Markov property from an underlying POMDP.
5. **Reward Clipping**: Environmental scores vary wildly across games (e.g., $+1$ in Pong vs $+10,000$ in Centipede). Clipping all positive rewards to $+1$ and negative rewards to $-1$ standardizes gradient norms across diverse games.

### Boundaries & Frontiers of DQN Applications
- **Pong**: Fully solvable by standard DQN within $10^6$ steps. Score bounds: $[-21, +21]$.
- **Breakout**: Requires Dueling DQN and frame stacking to discover the tunneling exploit (digging a channel to bounce the ball behind the wall).
- **Montezuma's Revenge / Pitfall**: Extreme failure cases for standard DQN. Sparse rewards combined with lethal traps require intrinsic exploration (RND, Go-Explore) or hierarchical planning.
- **Pokémon Red & Minecraft Frontiers**: Complex long-horizon tasks (over $10^5$ steps) with combinatorial state spaces require external memory, goal-conditioned architectures, or hierarchical semi-MDP formulations.


# 4. Chapter 08: Policy Gradients & REINFORCE

### Sources
- `docs/en/chapter08_policy_gradient/pg-necessity.md`
- `docs/en/chapter08_policy_gradient/reinforce.md`
- `docs/en/chapter08_policy_gradient/pg-improvements.md`
- `docs/en/chapter08_policy_gradient/dice-game.md`
- `docs/en/chapter08_policy_gradient/cartpole.md`
- `docs/en/chapter08_policy_gradient/baseline-experiment.md`
- `docs/en/chapter08_policy_gradient/cartpole-baseline.md`

---

## 4.1 Why Policy Gradients: Value-Based Breakdown & Action Distribution

Value-based algorithms (such as Q-Learning and DQN) learn an action-value function $Q(s, a)$ and derive the acting policy via greedy maximization:
$$\pi(s) = \arg\max_{a \in \mathcal{A}} Q(s, a)$$
While effective for discrete domains with few actions, this value-to-policy mapping breaks down in several critical domains:

### 1. Continuous Action Spaces ($\mathcal{A} \subseteq \mathbb{R}^d$)
In robotic joint control, autonomous driving steering, or physics simulators, actions are continuous real-valued vectors. Computing $\arg\max_{a} Q(s, a)$ requires solving an internal non-linear, non-convex global optimization problem at every single timestep. Discretizing continuous dimensions suffers from exponential combinatorial explosion: discretizing a 6-joint robotic arm with 100 bins per joint produces $100^6 = 10^{12}$ actions. Evaluating a neural network over $10^{12}$ choices at $1\mu\text{s}$ per evaluation requires $10^6\text{ seconds} \approx 11.6\text{ days}$ for a single control step.

### 2. Large Discrete Action Spaces & Language Generation
In Large Language Model (LLM) generation, the action space is the token vocabulary $|\mathcal{A}| \approx 32,000 - 128,000$. Purely greedy selection over Q-values produces repetitive, degenerated text and destroys sampling diversity. LLMs require sampling from a smooth, normalized probability distribution $\pi_\theta(a \mid s)$.

### 3. Stochastic Optimal Policies (POMDPs & Game Theory)
In partially observable environments (POMDPs) or competitive multi-agent games with imperfect information (e.g., Poker, Rock-Paper-Scissors), the game-theoretic Nash equilibrium requires a stochastic policy (e.g., uniform random $\frac{1}{3}$ for Rock-Paper-Scissors). Deterministic value-based policies can be completely exploited by adversaries.

### 4. Smooth Optimization Manifolds
In value-based methods, an infinitesimal adjustment in parameter $\theta$ can cause a discontinuous jump in $\arg\max_a Q(s, a)$, creating abrupt policy instability. Policy gradient parameterizes $\pi_\theta(a \mid s)$ directly with smooth neural network layers, guaranteeing that small parameter changes produce continuous, bounded shifts in action probabilities.

---

## 4.2 The Policy Objective Formulation $J(\theta)$

Let a trajectory $\tau$ be a sequence of states, actions, and rewards:
$$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots, s_T)$$
The trajectory probability density under policy $\pi_\theta$ and environment dynamics $P$ is:
$$P(\tau; \theta) = \rho_0(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t \mid s_t) P(s_{t+1} \mid s_t, a_t)$$
The objective of reinforcement learning is to find policy parameters $\theta^*$ that maximize the expected discounted cumulative return:
$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ R(\tau) \right] = \int P(\tau; \theta) R(\tau) d\tau$$
Where $R(\tau) = \sum_{t=0}^{T-1} \gamma^t r_t$ is the total trajectory return. Optimization proceeds via gradient ascent:
$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$

---

## 4.3 The Policy Gradient Theorem: Analytic Derivation & Score Function Trick

Computing $\nabla_\theta J(\theta)$ directly is challenging because the expectation distribution $P(\tau; \theta)$ itself depends on the optimization parameter $\theta$.

### The Log-Derivative Trick (Likelihood Ratio / Score Function)
From elementary calculus:
$$\nabla_\theta f(\theta) = f(\theta) \frac{\nabla_\theta f(\theta)}{f(\theta)} = f(\theta) \nabla_\theta \log f(\theta)$$
Applying this identity to the trajectory probability density $P(\tau; \theta)$:
$$\nabla_\theta P(\tau; \theta) = P(\tau; \theta) \nabla_\theta \log P(\tau; \theta)$$

### Full Analytical Derivation
$$\nabla_\theta J(\theta) = \nabla_\theta \int P(\tau; \theta) R(\tau) d\tau$$
Exchanging differentiation and integration (via Leibniz integral rule under standard regularity conditions):
$$= \int \nabla_\theta P(\tau; \theta) R(\tau) d\tau$$
Applying the log-derivative trick:
$$= \int P(\tau; \theta) \nabla_\theta \log P(\tau; \theta) R(\tau) d\tau$$
Converting back to mathematical expectation:
$$= \mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log P(\tau; \theta) R(\tau) \right]$$

### Elimination of Environmental Dynamics
Expanding the natural logarithm of trajectory probability $P(\tau; \theta)$:
$$\log P(\tau; \theta) = \log \rho_0(s_0) + \sum_{t=0}^{T-1} \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T-1} \log P(s_{t+1} \mid s_t, a_t)$$
Taking the gradient with respect to policy parameter vector $\theta$:
$$\nabla_\theta \log P(\tau; \theta) = \nabla_\theta \log \rho_0(s_0) + \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) + \sum_{t=0}^{T-1} \nabla_\theta \log P(s_{t+1} \mid s_t, a_t)$$
Because initial state distribution $\rho_0(s_0)$ and transition dynamics $P(s_{t+1} \mid s_t, a_t)$ are properties of the environment and completely independent of $\theta$:
$$\nabla_\theta \log \rho_0(s_0) = \mathbf{0}, \quad \nabla_\theta \log P(s_{t+1} \mid s_t, a_t) = \mathbf{0}$$
$$\nabla_\theta \log P(\tau; \theta) = \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t)$$

### Policy Gradient Theorem (Analytic Form)
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) R(\tau) \right]$$
*Engineering Significance*: This fundamental result proves that we can optimize policy parameters $\theta$ via gradient ascent **without having any model or knowledge of the environmental transition probabilities $P(s_{t+1} \mid s_t, a_t)$**.

---

## 4.4 Causality, Temporal Independence, & Reward-to-Go

The naive policy gradient multiplies every step's score function $\nabla_\theta \log \pi_\theta(a_t \mid s_t)$ by the total trajectory return $R(\tau) = \sum_{k=0}^{T-1} \gamma^k r_k$. This violates temporal causality: an action $a_t$ chosen at time $t$ cannot retroactively affect rewards received in past timesteps $t' < t$.

### Causality Proof
Expand the return into past and future terms:
$$\mathbb{E}_{\tau} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) R(\tau) \right] = \mathbb{E}_{\tau} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \left( \sum_{t'=0}^{t-1} \gamma^{t'} r_{t'} + \sum_{k=0}^{T - t - 1} \gamma^{t+k} r_{t+k} \right) \right]$$
Examining the expectation of the past reward terms via conditional expectation conditioned on history up to time $t$:
$$\mathbb{E}_{\tau} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \sum_{t'=0}^{t-1} \gamma^{t'} r_{t'} \right] = \mathbb{E}_{s_0:t, a_0:t-1} \left[ \left( \sum_{t'=0}^{t-1} \gamma^{t'} r_{t'} \right) \mathbb{E}_{a_t \sim \pi_\theta(\cdot \mid s_t)} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \;\middle|\; s_t \right] \right]$$
Evaluating the inner expectation across action space:
$$\mathbb{E}_{a_t \sim \pi_\theta(\cdot \mid s_t)} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right] = \sum_{a \in \mathcal{A}} \pi_\theta(a \mid s_t) \frac{\nabla_\theta \pi_\theta(a \mid s_t)}{\pi_\theta(a \mid s_t)} = \nabla_\theta \sum_{a \in \mathcal{A}} \pi_\theta(a \mid s_t) = \nabla_\theta (1) = \mathbf{0}$$
Because the inner expectation vanishes identically, past rewards have zero correlation with the gradient score function.

### Reward-to-Go Formulation
Dropping past rewards yields the causal Policy Gradient:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) G_t \right]$$
Where $G_t$ is the discounted reward-to-go from timestep $t$ onward:
$$G_t = \sum_{k=0}^{T - t - 1} \gamma^k r_{t+k}$$
*Variance Reduction*: Removing past random variables eliminates accumulated past noise, significantly reducing the variance of the gradient estimate while strictly maintaining zero estimation bias.

---

## 4.5 Baseline Invariance Proof & Variance Reduction

Even with reward-to-go, the empirical return $G_t$ varies wildly across trajectories, causing high gradient variance. We can subtract an arbitrary state-dependent baseline $b(s_t)$ from $G_t$:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \left( G_t - b(s_t) \right) \right]$$

### Analytical Theorem: Baseline Invariance
For any baseline function $b(s_t)$ that is independent of action $a_t$:
$$\mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T-1} \nabla_\theta \log \pi_\theta(a_t \mid s_t) b(s_t) \right] = \mathbf{0}$$

### Mathematical Proof
Applying the tower property of conditional expectation:
$$\mathbb{E}_{s_t, a_t} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) b(s_t) \right] = \mathbb{E}_{s_t} \left[ b(s_t) \mathbb{E}_{a_t \sim \pi_\theta(\cdot \mid s_t)} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \;\middle|\; s_t \right] \right]$$
Expanding the inner expectation over action distribution:
$$\mathbb{E}_{a_t \sim \pi_\theta(\cdot \mid s_t)} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \right] = \sum_{a \in \mathcal{A}} \pi_\theta(a \mid s_t) \frac{\nabla_\theta \pi_\theta(a \mid s_t)}{\pi_\theta(a \mid s_t)} = \sum_{a \in \mathcal{A}} \nabla_\theta \pi_\theta(a \mid s_t) = \nabla_\theta \sum_{a \in \mathcal{A}} \pi_\theta(a \mid s_t)$$
Because policy distribution is normalized ($\sum_{a \in \mathcal{A}} \pi_\theta(a \mid s_t) = 1$):
$$= \nabla_\theta (1) = \mathbf{0}$$
Multiplying by $b(s_t)$ and integrating over state distribution:
$$\mathbb{E}_{s_t} \left[ b(s_t) \cdot \mathbf{0} \right] = \mathbf{0} \quad \blacksquare$$

### The Optimal Value Baseline $V(s_t)$
Subtracting $b(s_t)$ preserves the expected gradient perfectly (zero bias). The variance of the gradient estimator $\text{Var}(\nabla_\theta \log \pi_\theta(a_t \mid s_t) (G_t - b(s_t)))$ is minimized when:
$$b^*(s_t) = \frac{\mathbb{E}[ \lVert \nabla_\theta \log \pi_\theta(a_t \mid s_t) \rVert_2^2 G_t \mid s_t ]}{\mathbb{E}[ \lVert \nabla_\theta \log \pi_\theta(a_t \mid s_t) \rVert_2^2 \mid s_t ]} \approx \mathbb{E}[G_t \mid s_t] = V^\pi(s_t)$$
Setting $b(s_t) = V^\pi(s_t)$ produces the empirical Advantage function:
$$\hat{A}_t = G_t - V(s_t)$$
- If an action yields $G_t > V(s_t)$ ($\hat{A}_t > 0$), its log-probability is boosted.
- If an action yields $G_t < V(s_t)$ ($\hat{A}_t < 0$), its log-probability is penalized, even if the absolute return $G_t$ was positive.

---

## 4.6 The REINFORCE Algorithm & Trajectory Rollout Loop

REINFORCE (Williams 1992) is the foundational Monte Carlo policy gradient algorithm:

```
Algorithm: REINFORCE with Learned Value Baseline
========================================================================================
1: Initialize policy network pi_theta and value network V_phi with weights theta, phi
2: for episode = 1 to M do
3:     Collect trajectory tau = (s_0, a_0, r_0, ..., s_T) using policy pi_theta
4:     Initialize return accumulator G <- 0
5:     Initialize lists of returns G_t and advantages A_t
6:     // Backward recursion for exact returns:
7:     for t = T-1 down to 0 do
8:         G <- r_t + gamma * G
9:         G_t.prepend(G)
10:        A_t.prepend(G - V_phi(s_t))
11:    end for
12:    // Parameter updates:
13:    Compute Policy Loss:
14:        L_pi(theta) = - (1/T) * sum_{t=0}^{T-1} log pi_theta(a_t | s_t) * A_t
15:    Compute Value Loss:
16:        L_V(phi) = (1/T) * sum_{t=0}^{T-1} (V_phi(s_t) - G_t)^2
17:    Perform gradient descent step on theta: theta <- theta - alpha_theta * grad L_pi
18:    Perform gradient descent step on phi: phi <- phi - alpha_phi * grad L_V
19: end for
========================================================================================
```

### PyTorch Surrogate Loss Construction
In modern autograd engines (PyTorch), the gradient ascent update $\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t \mid s_t) A_t$ is implemented by minimizing a surrogate loss:
$$\mathcal{L}_{\text{surrogate}}(\theta) = - \frac{1}{|B|} \sum_{t \in B} \log \pi_\theta(a_t \mid s_t) \cdot \text{detach}(\hat{A}_t)$$
The `detach()` call on $\hat{A}_t$ is mandatory to prevent policy gradients from backpropagating into the value network parameters $\phi$.

---

## 4.7 Discrete vs. Continuous Policy Formulations

### Discrete Action Spaces
For $|\mathcal{A}| = K$, the policy network outputs an unnormalized logit vector $z_\theta(s) \in \mathbb{R}^K$. Action probabilities are computed via Softmax:
$$\pi_\theta(a_i \mid s) = \frac{e^{z_i(s)}}{\sum_{j=1}^K e^{z_j(s)}}$$
The log-probability is:
$$\log \pi_\theta(a_i \mid s) = z_i(s) - \log \sum_{j=1}^K e^{z_j(s)}$$
The gradient with respect to output logit $z_j$ is:
$$\frac{\partial \log \pi_\theta(a_i \mid s)}{\partial z_j} = \mathbf{1}[i = j] - \pi_\theta(a_j \mid s)$$

### Continuous Action Spaces
For continuous action space $\mathcal{A} \subseteq \mathbb{R}^d$, the policy network parameterizes a Gaussian distribution $\mathcal{N}(\mu_\theta(s), \Sigma_\theta(s))$. Assuming a diagonal covariance matrix with standard deviation vector $\sigma_\theta(s)$:
$$\pi_\theta(a \mid s) = \prod_{k=1}^d \frac{1}{\sqrt{2\pi}\sigma_k(s)} \exp\left( - \frac{(a_k - \mu_k(s))^2}{2 \sigma_k(s)^2} \right)$$
The log-probability density is:
$$\log \pi_\theta(a \mid s) = - \frac{1}{2} \sum_{k=1}^d \left[ \left( \frac{a_k - \mu_k(s)}{\sigma_k(s)} \right)^2 + 2 \log \sigma_k(s) + \log(2\pi) \right]$$
The gradients with respect to predicted mean $\mu_k$ and standard deviation $\sigma_k$ are:
$$\nabla_\mu \log \pi_\theta(a \mid s) = \frac{a_k - \mu_k(s)}{\sigma_k(s)^2}$$
$$\nabla_\sigma \log \pi_\theta(a \mid s) = \frac{(a_k - \mu_k(s))^2}{\sigma_k(s)^3} - \frac{1}{\sigma_k(s)}$$

---

## 4.8 Experimental Validations: Bandit Oscillations & CartPole Baselines

### 1. Two-Armed Dice Game Bandit
In an environment where Arm A pays $+1$ with probability $0.6$ and Arm B pays $+1$ with probability $0.4$:
- **REINFORCE without Baseline**: The gradient updates are noisy. Even when Arm A is chosen and wins ($G_0 = 1$), Arm B also occasionally wins ($G_0 = 1$), providing positive reinforcement to both actions. Because both actions receive positive gradients, their probabilities oscillate violently, delaying convergence by thousands of steps.
- **REINFORCE with Average Baseline ($b = 0.5$)**: When Arm A wins ($G_0 = 1$), advantage is $+0.5$; when it loses ($G_0 = 0$), advantage is $-0.5$. Actions that win less frequently than average receive negative reinforcement, immediately suppressing Arm B and accelerating convergence.

### 2. CartPole Baseline Comparison Experiment
In `CartPole-v1` experiments comparing REINFORCE with and without a learned neural value baseline $V_\phi(s)$:

| Experimental Metric | REINFORCE (No Baseline) | REINFORCE + Value Baseline ($V(s)$) |
| :--- | :--- | :--- |
| **Gradient Variance** | Extremely High ($\sigma^2 \sim 10^2 - 10^3$) | Substantially Lower ($\sigma^2 \sim 10^0 - 10^1$) |
| **Training Curve Stability** | Severe oscillations; sudden crashes to return $\sim 20$ | Smooth monotonic ascent |
| **Steps to Solve ($G = 500$)** | $\sim 80,000 - 120,000$ steps (inconsistent) | $\sim 30,000 - 45,000$ steps (reliable across seeds) |
| **Failure Modes** | Policy entropy prematurely collapses due to rare lucky runs | Stable entropy decay preserving exploration |


# 5. Chapter 09: Actor-Critic Architectures

### Sources
- `docs/en/chapter09_actor_critic/advantage-function.md`
- `docs/en/chapter09_actor_critic/actor-critic.md`
- `docs/en/chapter09_actor_critic/critic-training.md`
- `docs/en/chapter09_actor_critic/pendulum.md`
- `docs/en/chapter09_actor_critic/bipedalwalker.md`
- `docs/en/chapter09_actor_critic/alphago.md`
- `docs/en/chapter09_actor_critic/ac-frontier.md`

---

## 5.1 From Monte Carlo Policy Gradient to Actor-Critic

The REINFORCE algorithm evaluates policy performance using the empirical Monte Carlo return $G_t = \sum_{k=0}^{T - t - 1} \gamma^k r_{t+k}$:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau} \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \left( G_t - V_\phi(s_t) \right) \right]$$
While $G_t$ is an unbiased estimator of true state-action value $Q^\pi(s_t, a_t)$, relying on Monte Carlo rollouts introduces two severe practical limitations:
1. **Episodic Dependency**: The agent cannot update its weights until the entire episode terminates. In lifelong learning, continuing tasks, or long-horizon robotics, waiting for episode completion is either impossible or introduces prohibitive latency.
2. **Cumulative Trajectory Variance**: $G_t$ compounds the stochasticity of all subsequent policy decisions and state transitions until step $T$, resulting in noisy gradient signals.

### The Actor-Critic Resolution
Actor-Critic resolves this by replacing the Monte Carlo return $G_t$ with a bootstrapped estimate provided by a learned **Critic** network $V_\phi(s)$ or $Q_\phi(s, a)$.
- **Actor ($\pi_\theta$)**: Parameterizes the decision-making policy distribution $\pi_\theta(a \mid s)$, updating parameters in the direction suggested by the Critic.
- **Critic ($V_\phi$)**: Evaluates states by learning the value function $V^\pi(s)$ via Temporal Difference (TD) learning, providing immediate, step-by-step feedback to the Actor.

---

## 5.2 The Actor-Critic Bias-Variance Tradeoff

The shift from REINFORCE to Actor-Critic represents a fundamental shift in the bias-variance spectrum:

| Dimension | REINFORCE (Monte Carlo Return $G_t$) | One-Step Actor-Critic (TD Error $\delta_t$) |
| :--- | :--- | :--- |
| **Target Estimator** | Realized cumulative return $G_t = \sum \gamma^k r_{t+k}$ | One-step bootstrapped target $r_t + \gamma V_\phi(s_{t+1})$ |
| **Estimation Bias** | **Zero Bias**: $\mathbb{E}[G_t \mid s_t, a_t] = Q^\pi(s_t, a_t)$ | **Biased**: Depends on accuracy of critic weights $\phi$ |
| **Gradient Variance** | **High Variance**: Multiplies across full future trajectory | **Low Variance**: Single-step transition stochasticity only |
| **Update Regime** | Offline / End-of-episode batch update | **Online / Step-by-step immediate update** |
| **Applicability** | Finite episodic MDPs only | Continuing, infinite-horizon, and episodic MDPs |
| **Convergence Rate** | Slow due to gradient noise | Faster initial learning; convergence depends on critic fit |

As training progresses and the Critic network converges toward the true value function $V^\pi(s)$, the estimation bias $\lVert V_\phi(s) - V^\pi(s) \rVert$ approaches zero, yielding a low-bias, low-variance gradient estimator.

---

## 5.3 Advantage Function Estimation via One-Step TD Error

Recall the Advantage function definition:
$$A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)$$
By the Bellman expectation equation:
$$Q^\pi(s_t, a_t) = \mathbb{E}_{s_{t+1}} \left[ r_t + \gamma V^\pi(s_{t+1}) \;\middle|\; s_t, a_t \right]$$
Taking a single sample transition $(s_t, a_t, r_t, s_{t+1})$, the quantity:
$$r_t + \gamma V_\phi(s_{t+1})$$
is an unbiased one-step sample estimate of $Q^\pi(s_t, a_t)$. Substituting this into the Advantage function yields the **one-step Temporal Difference (TD) Error**:
$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

### Dual Role of the TD Error
The single scalar quantity $\delta_t$ simultaneously drives both networks:
1. **Critic Training**: $\delta_t$ is the prediction residual of the Critic. The Critic minimizes Mean Squared TD Error:
   $$\mathcal{L}_{\text{critic}}(\phi) = \delta_t^2 = \left( r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t) \right)^2$$
   Gradient descent update on Critic:
   $$\phi \leftarrow \phi + \alpha_\phi \delta_t \nabla_\phi V_\phi(s_t)$$
2. **Actor Training**: $\delta_t$ acts as the empirical Advantage $\hat{A}(s_t, a_t)$. The Actor performs policy gradient ascent:
   $$\theta \leftarrow \theta + \alpha_\theta \nabla_\theta \log \pi_\theta(a_t \mid s_t) \delta_t$$

---

## 5.4 Network Architecture: Shared Backbone vs. Disjoint Parameterization

In deep Actor-Critic implementations, two primary topological designs exist:

```
[Design A: Shared Feature Trunk]             [Design B: Disjoint Networks]

          Input State s                                 Input State s
               |                                       /             \
               v                                      v               v
     +-------------------+                     +-------------+ +-------------+
     |   Shared Trunk    |                     | Actor Trunk | | Critic Trunk|
     | (Conv / MLP) phi  |                     |   theta     | |     phi     |
     +-------------------+                     +-------------+ +-------------+
           /       \                                  |               |
          v         v                                 v               v
    +----------+ +----------+                   +----------+   +----------+
    |  Actor   | |  Critic  |                   |  Actor   |   |  Critic  |
    | Head pi  | |  Head V  |                   | Head pi  |   |  Head V  |
    +----------+ +----------+                   +----------+   +----------+
```

### Comparative Analysis
1. **Shared Feature Trunk (Design A)**:
   - *Advantage*: Compact parameter count; representations learned from value regression enrich policy feature extraction (crucial in visual domains like Atari).
   - *Risk (Gradient Interference)*: The Critic loss $\mathcal{L}_V = \delta^2$ frequently produces gradient magnitudes orders of magnitude larger than policy gradient $\nabla_\theta \mathcal{L}_\pi$. Unchecked, value gradients dominate the shared trunk, destabilizing policy representations.
   - *Mitigation*: Introduce loss scale weighting:
     $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{actor}} + c_1 \mathcal{L}_{\text{critic}} - c_2 \mathcal{H}(\pi)$$
     with $c_1 \in [0.5, 1.0]$ and clip gradient norms (`max_grad_norm = 0.5`).
2. **Disjoint Networks (Design B)**:
   - *Advantage*: Complete numerical isolation. Zero gradient interference between policy and value objectives. Actor and Critic can employ different optimizers and distinct learning rates (e.g., $\alpha_{\text{critic}} = 10^{-3}$, $\alpha_{\text{actor}} = 3 \times 10^{-4}$).
   - *Disadvantage*: Doubles forward-pass computation and memory usage.

---

## 5.5 Critic Training Mechanics: DP vs. MC vs. TD Walkthrough

Consider a canonical 3-state Markov chain:
$$S \xrightarrow{r_1} M \xrightarrow{r_2} G \quad \text{(terminal Goal)}$$
Initial critic estimates: $V_0(S) = 0, V_0(M) = 0, V_0(G) = 0$. Discount factor $\gamma = 1.0$, learning rate $\alpha = 0.5$.
Trajectory experienced: $S \xrightarrow{-2} S \xrightarrow{-1} M \xrightarrow{-2} S \xrightarrow{-1} M \xrightarrow{-1} G$. Realized total return from start is $G_0 = -7.0$.

### 1. Dynamic Programming (DP) Value Backup
Requires transition matrix $P$ and reward function $R$. Iteratively applies the expectation Bellman operator:
$$V_{k+1}(s) \leftarrow \sum_a \pi(a \mid s) \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V_k(s') \right]$$
Full knowledge of transition branches guarantees monotonic contraction to true values without sampling variance.

### 2. Monte Carlo (MC) Critic Update
Waits until episode completion, then computes true realized return $G_t$:
$$G_0 = -7.0 \implies \delta_{\text{MC}} = G_0 - V_0(S) = -7.0 - 0 = -7.0$$
$$V_1(S) \leftarrow V_0(S) + \alpha \delta_{\text{MC}} = 0 + 0.5(-7.0) = -3.5$$
*Takeaway*: Updates are large, direct, and unbiased, but fail if the episode never terminates.

### 3. Temporal Difference (TD(0)) Critic Update
Updates state values sequentially at every transition step:
- *Step 1 ($S \xrightarrow{-2} S$)*:
  $$\delta = r + \gamma V_0(S) - V_0(S) = -2 + 1(0) - 0 = -2.0 \implies V_1(S) \leftarrow 0 + 0.5(-2.0) = -1.0$$
- *Step 2 ($S \xrightarrow{-1} M$)*:
  $$\delta = r + \gamma V_0(M) - V_1(S) = -1 + 1(0) - (-1.0) = 0.0 \implies V_2(S) \leftarrow -1.0 + 0.5(0) = -1.0$$
- *Step 3 ($M \xrightarrow{-2} S$)*:
  $$\delta = r + \gamma V_2(S) - V_0(M) = -2 + 1(-1.0) - 0 = -3.0 \implies V_1(M) \leftarrow 0 + 0.5(-3.0) = -1.5$$
*Takeaway*: Information propagates backward incrementally. The critic bootstraps from existing values, enabling immediate local updates.

---

## 5.6 Continuous Action Control: Gaussian Policy Parameterization

In continuous control problems ($\mathcal{A} = \mathbb{R}^d$), the Actor network outputs the parameters of a multivariate Gaussian distribution. For diagonal covariance:
$$a \sim \pi_\theta(\cdot \mid s) = \mathcal{N}\left( \mu_\theta(s), \Sigma_\theta(s) \right), \quad \Sigma_\theta(s) = \text{diag}(\sigma_1^2(s), \ldots, \sigma_d^2(s))$$

### Network Output Heads
- **Mean Head ($\mu_\theta(s)$)**: Linear layer with optional $\tanh$ activation scaled to action bounds $[a_{\text{low}}, a_{\text{high}}]$.
- **Standard Deviation Head**: Parameterized either as state-dependent output $\log \sigma_\theta(s)$ or as a state-independent trainable parameter vector $\log \sigma \in \mathbb{R}^d$. Log-scale parameterization guarantees strictly positive standard deviations: $\sigma = \exp(\log \sigma)$.

### Policy Entropy for Continuous Gaussian Actions
$$H(\pi_\theta(\cdot \mid s)) = \frac{1}{2} \sum_{i=1}^d \log\left( 2\pi e \sigma_i^2(s) \right)$$
Maximizing entropy encourages wide exploration ($\sigma$ remains large); as confidence increases, $\sigma$ contracts naturally.

---

## 5.7 Complex Control Benchmark: Pendulum-v1 & BipedalWalker-v3

### 1. Inverted Pendulum Torque Control (`Pendulum-v1`)
- **State Space**: $s = [\cos\theta, \sin\theta, \dot{\theta}] \in \mathbb{R}^3$.
- **Action Space**: Continuous torque $a \in [-2.0, 2.0]\text{ N}\cdot\text{m}$.
- **Reward Dynamics**: Penalizes deviation from upright vertical orientation ($\theta = 0$), angular velocity, and control effort:
  $$r_t = - \left( \theta^2 + 0.1 \dot{\theta}^2 + 0.001 a_t^2 \right)$$
  Where angle $\theta \in [-\pi, \pi]$ (with $\theta = 0$ vertical upright). Maximum possible step reward is $0.0$; typical failing runs yield $-1600$ over 200 steps.
- **Physical Dynamics & Actor Strategy**: The motor torque ($2.0\text{ N}\cdot\text{m}$) is insufficient to swing the pendulum upright directly against gravity from a dead hang. The Actor must discover a non-greedy multi-step strategy: swing backward to build gravitational kinetic energy, then apply maximum forward torque to swing over the top, and finally decelerate precisely to balance at upright equilibrium ($\theta = 0, \dot{\theta} = 0, a = 0$).

### 2. Complex Planar Locomotion (`BipedalWalker-v3`)
- **State Space**: $\mathcal{S} \subseteq \mathbb{R}^{24}$ continuous vector containing:
  - Hull angle, angular velocity, horizontal speed, vertical speed, leg positions, joint velocities, ground contact sensors, and 10 lidar rangefinder depth scans.
- **Action Space**: $\mathcal{A} \subseteq [-1.0, 1.0]^4$ continuous torques applied to 4 joints: Hip 1, Knee 1, Hip 2, Knee 2.
- **Reward Dynamics**: Forward progress reward $+r_{\text{forward}}$, energy penalty $-0.00035 \sum a_i^2$, and catastrophic fall penalty $-100.0$.
- **Three-Stage Learning Progression**:
  1. *Early Phase ($0 - 500\text{k}$ steps, Return $\sim -50$)*: Learns basic upright balance. Lander drags rear leg to prevent catastrophic falls.
  2. *Mid Phase ($500\text{k} - 2\text{M}$ steps, Return $\sim +100 - +200$)*: Discovers rhythmic leg alternation; gaits are uncoordinated, with frequent stumbles.
  3. *Late Phase ($2\text{M} - 3\text{M}$ steps, Return $\ge +270$)*: Learns smooth, energy-efficient bipedal locomotion, actively modulating joint torques across uneven terrain.
- **A2C vs. PPO Stability Comparison**: Vanilla Advantage Actor-Critic (A2C) exhibits high fragility on BipedalWalker: an unexpected trip or collision causes a large TD error that triggers an excessive policy update step, collapsing the walking policy into a catastrophic falling loop. PPO's clipped objective enforces a strict trust region, guaranteeing steady monotonic improvement.

---

## 5.8 Large-Scale Frontiers: AlphaGo, AlphaStar, SAC, & NVIDIA Isaac Lab

Actor-Critic architectures form the computational core of breakthrough achievements in artificial intelligence:

### 1. AlphaGo & AlphaGo Zero (Silver et al., Nature 2016, 2017)
- **Actor-Critic Hybridization**: Combined deep convolutional policy network $\pi_\theta(a \mid s)$ (19x19 Go board) and value network $v_\phi(s) \in [-1, +1]$ with Monte Carlo Tree Search (MCTS).
- **PUCT Node Selection Formula**:
  $$\text{PUCT}(s, a) = Q(s, a) + c_{\text{puct}} P(s, a) \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$$
  Where $P(s, a) = \pi_\theta(a \mid s)$ provides prior probability from the policy network, $Q(s, a)$ is the mean evaluation from MCTS rollouts and value network predictions, and $N(s, a)$ is visit count.
- **AlphaGo Zero Innovation**: Eliminated all human expert game records. Trained entirely via reinforcement learning self-play:
  $$\mathcal{L} = (z - v_\phi(s))^2 - \boldsymbol{\pi}_{\text{MCTS}}^T \log \boldsymbol{p}_\theta(s) + c \lVert \theta \rVert^2$$
  Where $z \in \{-1, +1\}$ is the self-play game winner, and $\boldsymbol{\pi}_{\text{MCTS}}$ is the search distribution.

### 2. AlphaStar: StarCraft II Grandmaster (Vinyals et al., Nature 2019)
- **Domain Complexity**: Real-time strategy with imperfect information (fog of war), combinatorial action spaces (unit selection, coordinates, abilities), and 10,000+ step horizons.
- **V-trace Off-Policy Actor-Critic**: Corrects for multi-step off-policy discrepancy between asynchronous rollout workers and central learner:
  $$v_s = V(s) + \sum_{t=s}^{s+n-1} \gamma^{t-s} \left( \prod_{k=s}^{t-1} c_k \right) \rho_t \left( r_t + \gamma V(s_{t+1}) - V(s_t) \right)$$
  Truncated importance sampling ratios:
  $$\rho_t = \min\left( \bar{\rho}, \frac{\pi(a_t \mid s_t)}{\mu(a_t \mid s_t)} \right), \quad c_k = \min\left( \bar{c}, \frac{\pi(a_k \mid s_k)}{\mu(a_k \mid s_k)} \right)$$
- **League Training Architecture**: Main Agents, Main Exploiters, and League Exploiters play continuously against current and historical checkpoints to prevent cyclic, non-transitive exploit loops (e.g., Rock-Paper-Scissors meta dynamics).

### 3. Soft Actor-Critic (SAC - Haarnoja et al., 2018)
Brings actor-critic from simulation to real physical robots by optimizing a maximum entropy objective, utilizing twin Q-critics to eliminate overestimation bias.

### 4. NVIDIA Isaac Lab (Industrial GPU Vectorized Robotics)
- **Massively Parallel Vectorization**: Executes $4,096$ to $16,384$ complete robot physics simulations concurrently on a single NVIDIA GPU using PhysX tensor cores, collecting tens of millions of environment steps per hour.
- **Domain Randomization for Sim-to-Real**: Randomizes physics properties (ground friction $\mu \sim \mathcal{U}(0.2, 1.2)$, robot mass $\pm 15\%$, joint damping, motor latency, sensor noise) during parallel training. The resulting Actor-Critic policy transfers directly to physical quadruped and humanoid robots without fine-tuning.


# 6. Chapter 10: Proximal Policy Optimization (PPO)

### Sources
- `docs/en/chapter10_ppo/ppo-clip-objective.md`
- `docs/en/chapter10_ppo/trust-region-clipping.md`
- `docs/en/chapter10_ppo/gae-reward-model.md`
- `docs/en/chapter10_ppo/ppo-math.md`
- `docs/en/chapter10_ppo/ppo-bipedal-walker.md`
- `docs/en/chapter10_ppo/ppo-game-benchmark.md`
- `docs/en/chapter10_ppo/rl-long-horizon-planning.md`

---

## 6.1 Vanilla Policy Gradient Failure: Step Size Sensitivity & Collapse

In supervised deep learning, choosing an overly aggressive learning rate causes a temporary spike in loss; subsequent mini-batches drawn from a static dataset allow the network to recover. In reinforcement learning, however, **the policy itself collects the training data**:
1. **Destructive Policy Shifts**: If a gradient update takes an excessively large step in parameter space, the policy can instantly degenerate into a dysfunctional state (e.g., repeatedly firing an engine until spinning out, or repeatedly choosing a single token).
2. **The Policy Collapse Trap**: Once the policy degenerates, all future rollouts collected by this policy will be catastrophic failures with near-zero or negative returns. The agent cannot collect informative transitions to compute correcting gradients, leading to permanent policy collapse.
3. **Sample Inefficiency**: Vanilla Policy Gradient is strictly on-policy. A batch of $N$ rollout transitions can be used for exactly one gradient update before being discarded, requiring tens of millions of environment interactions.

---

## 6.2 Importance Sampling & Off-Policy Trajectory Reuse

To reuse rollout trajectories across multiple mini-batch optimization epochs without re-simulating the environment, PPO applies **Importance Sampling**:

### Mathematical Identity
Given expectation under distribution $p(x)$, we can evaluate it under an alternative sampling distribution $q(x)$ provided $q(x) > 0$ whenever $p(x) > 0$:
$$\mathbb{E}_{x \sim p}[f(x)] = \int p(x) f(x) dx = \int q(x) \frac{p(x)}{q(x)} f(x) dx = \mathbb{E}_{x \sim q}\left[ \frac{p(x)}{q(x)} f(x) \right]$$
Where $\frac{p(x)}{q(x)}$ is the importance sampling ratio.

### Probability Ratio Formulation
Let $\pi_{\theta_{\text{old}}}$ denote the behavioral policy snapshot that collected the rollout batch, and $\pi_\theta$ denote the current optimizing policy:
$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$
- At the start of an update epoch, $\theta = \theta_{\text{old}}$, so $r_t(\theta) = 1.0$.
- As $\theta$ updates, $r_t(\theta)$ departs from $1.0$.

### Conservative Policy Iteration (CPI) Surrogate Objective
$$L^{\text{CPI}}(\theta) = \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t \right] = \hat{\mathbb{E}}_t \left[ r_t(\theta) \hat{A}_t \right]$$
Taking the gradient of $L^{\text{CPI}}(\theta)$ with respect to $\theta$ at the initial point $\theta = \theta_{\text{old}}$:
$$\left. \nabla_\theta L^{\text{CPI}}(\theta) \right|_{\theta = \theta_{\text{old}}} = \left. \hat{\mathbb{E}}_t \left[ \frac{\nabla_\theta \pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} \hat{A}_t \right] \right|_{\theta = \theta_{\text{old}}} = \hat{\mathbb{E}}_t \left[ \frac{\nabla_\theta \pi_\theta(a_t \mid s_t)}{\pi_\theta(a_t \mid s_t)} \hat{A}_t \right] = \hat{\mathbb{E}}_t \left[ \nabla_\theta \log \pi_\theta(a_t \mid s_t) \hat{A}_t \right]$$
The gradient of the surrogate objective at $\theta = \theta_{\text{old}}$ is identical to the on-policy policy gradient.

---

## 6.3 Trust Region Policy Optimization (TRPO) & Second-Order Constraints

Maximizing $L^{\text{CPI}}(\theta)$ without constraints causes $r_t(\theta)$ to explode, producing catastrophic policy shifts. TRPO (Schulman et al., 2015) bounds policy divergence using a hard Kullback-Leibler (KL) divergence trust region:
$$\max_\theta \hat{\mathbb{E}}_t \left[ r_t(\theta) \hat{A}_t \right] \quad \text{subject to } \hat{\mathbb{E}}_t \left[ D_{\text{KL}}(\pi_{\theta_{\text{old}}}(\cdot \mid s_t) \parallel \pi_\theta(\cdot \mid s_t)) \right] \le \delta$$

### Second-Order TRPO Mechanics
Using Taylor series expansions:
- Objective linear approximation: $L^{\text{CPI}}(\theta) \approx g^T (\theta - \theta_{\text{old}})$ where $g = \nabla_\theta L^{\text{CPI}}$.
- KL divergence quadratic approximation: $D_{\text{KL}} \approx \frac{1}{2} (\theta - \theta_{\text{old}})^T F (\theta - \theta_{\text{old}})$, where $F = \nabla_\theta^2 D_{\text{KL}}$ is the **Fisher Information Matrix (FIM)**.

The constrained quadratic program has the analytical solution:
$$\theta - \theta_{\text{old}} = \sqrt{\frac{2\delta}{g^T F^{-1} g}} F^{-1} g$$
*Computational Bottlenecks of TRPO*:
1. Computing $F^{-1} g$ for a neural network with $M$ parameters requires inverting an $M \times M$ matrix ($\mathcal{O}(M^3)$), requiring Conjugate Gradient approximations.
2. Incompatible with parameter sharing between Actor and Critic networks.
3. Complex to implement, tune, and scale across distributed architectures.

---

## 6.4 PPO Clipped Surrogate Objective: Full Analytical Derivation

Proximal Policy Optimization (Schulman et al., 2017) emulates the trust region dynamics of TRPO using a first-order clipped surrogate objective:
$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t \right) \right]$$
Where clipping hyperparameter $\epsilon \in [0.1, 0.3]$ (typically $\epsilon = 0.2$).

```
        Case 1: Advantage A_t > 0                     Case 2: Advantage A_t < 0
          (Action is Better)                             (Action is Worse)
   L^CLIP                                         L^CLIP
     ^                                              ^
     |             (1+eps) * A_t                    |
     |         +------------------                  |                (1+eps) * A_t
     |        /                                     |               /
     |       /                                      |              /
     |      /  r_t * A_t                            |  r_t * A_t  /
     |     /                                        |            /
     |    /                                         +-----------+
     |   /                                          | (1-eps)*A_t
     +--+-------+-----------------> r_t             +-----------+----+------------> r_t
        0      1-eps   1   1+eps                                0   1-eps 1  1+eps
```

### Rigorous Piecewise Analysis

#### Case 1: Positive Advantage ($\hat{A}_t > 0$)
The action yielded higher returns than expected. We want to increase its probability ($r_t(\theta) > 1$).
$$\text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t = \begin{cases} (1 - \epsilon)\hat{A}_t & \text{if } r_t < 1 - \epsilon \\ r_t(\theta)\hat{A}_t & \text{if } 1 - \epsilon \le r_t \le 1 + \epsilon \\ (1 + \epsilon)\hat{A}_t & \text{if } r_t > 1 + \epsilon \end{cases}$$
Taking the minimum with $r_t(\theta)\hat{A}_t$:
$$L^{\text{CLIP}}(\theta) = \begin{cases} r_t(\theta)\hat{A}_t & \text{if } r_t(\theta) \le 1 + \epsilon \\ (1 + \epsilon)\hat{A}_t & \text{if } r_t(\theta) > 1 + \epsilon \end{cases}$$
- **When $r_t(\theta) \le 1 + \epsilon$**: The gradient is active: $\nabla_\theta L^{\text{CLIP}} = \hat{A}_t \nabla_\theta r_t(\theta) > 0$. The optimizer increases the probability of action $a_t$.
- **When $r_t(\theta) > 1 + \epsilon$**: The objective hits a plateau $(1 + \epsilon)\hat{A}_t$. The derivative is zero: $\nabla_\theta L^{\text{CLIP}} = \mathbf{0}$. The optimizer stops updating $\theta$ for this sample, preventing policy drift beyond the trust region.

#### Case 2: Negative Advantage ($\hat{A}_t < 0$)
The action yielded lower returns than expected. We want to decrease its probability ($r_t(\theta) < 1$).
*Note*: Multiplying by negative $\hat{A}_t$ reverses all inequality directions:
$$\text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon) \hat{A}_t = \begin{cases} (1 + \epsilon)\hat{A}_t & \text{if } r_t > 1 + \epsilon \\ r_t(\theta)\hat{A}_t & \text{if } 1 - \epsilon \le r_t \le 1 + \epsilon \\ (1 - \epsilon)\hat{A}_t & \text{if } r_t < 1 - \epsilon \end{cases}$$
Taking the minimum with $r_t(\theta)\hat{A}_t$:
$$L^{\text{CLIP}}(\theta) = \begin{cases} (1 - \epsilon)\hat{A}_t & \text{if } r_t(\theta) < 1 - \epsilon \\ r_t(\theta)\hat{A}_t & \text{if } r_t(\theta) \ge 1 - \epsilon \end{cases}$$
- **When $r_t(\theta) \ge 1 - \epsilon$**: The gradient is active, driving down the probability of action $a_t$.
- **When $r_t(\theta) < 1 - \epsilon$**: The objective hits a ceiling $(1 - \epsilon)\hat{A}_t$. The derivative vanishes: $\nabla_\theta L^{\text{CLIP}} = \mathbf{0}$. This prevents over-penalizing bad actions, protecting the policy against numerical collapse.

### Why the Minimum Operator is Crucial
The $\min$ creates a **pessimistic lower bound**:
$$L^{\text{CLIP}}(\theta) \le L^{\text{CPI}}(\theta)$$
Clipping ignores changes in the objective when the probability ratio moves in a direction that would make the objective overly optimistic.

---

## 6.5 Generalized Advantage Estimation (GAE): Derivation & Recursive Form

Generalized Advantage Estimation (Schulman et al., 2016) constructs a controlled bias-variance tradeoff for advantage estimation.

### The Spectrum of Multi-Step TD Errors
Define the fundamental Temporal Difference residual:
$$\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$$
Examine multi-step advantage estimators:
$$\hat{A}_t^{(1)} = \delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$$
$$\hat{A}_t^{(2)} = \delta_t^V + \gamma \delta_{t+1}^V = r_t + \gamma r_{t+1} + \gamma^2 V(s_{t+2}) - V(s_t)$$
$$\hat{A}_t^{(k)} = \sum_{l=0}^{k-1} \gamma^l \delta_{t+l}^V = \sum_{l=0}^{k-1} \gamma^l r_{t+l} + \gamma^k V(s_{t+k}) - V(s_t)$$
$$\hat{A}_t^{(\infty)} = \sum_{l=0}^\infty \gamma^l \delta_{t+l}^V = \sum_{l=0}^\infty \gamma^l r_{t+l} - V(s_t) = G_t - V(s_t)$$

### GAE Formulation
GAE is the exponentially weighted moving average of all $k$-step advantage estimators, parameterized by $\lambda \in [0, 1]$:
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = (1 - \lambda) \sum_{k=1}^{\infty} \lambda^{k-1} \hat{A}_t^{(k)}$$
Substituting the expansion of $\hat{A}_t^{(k)}$:
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = (1 - \lambda) \left( \delta_t^V + \lambda (\delta_t^V + \gamma \delta_{t+1}^V) + \lambda^2 (\delta_t^V + \gamma \delta_{t+1}^V + \gamma^2 \delta_{t+2}^V) + \cdots \right)$$
Grouping terms by $\delta_{t+l}^V$:
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \delta_t^V \sum_{k=1}^\infty (1 - \lambda) \lambda^{k-1} + \gamma \delta_{t+1}^V \sum_{k=2}^\infty (1 - \lambda) \lambda^{k-1} + \cdots$$
Since $\sum_{k=m}^\infty (1 - \lambda)\lambda^{k-1} = \lambda^{m-1}$:
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}^V$$

### Backward Linear-Time Recursive Implementation
$$\hat{A}_t^{\text{GAE}} = \delta_t^V + \gamma \lambda (1 - d_t) \hat{A}_{t+1}^{\text{GAE}}$$
Where $d_t \in \{0, 1\}$ is the episode termination flag (`done`). This recursion computes advantages for an entire rollout of length $T$ in a single backward pass ($\mathcal{O}(T)$ operations).

### Target Return for Critic Training
The advantage estimates are added back to state values to form stable regression targets for the critic:
$$\hat{R}_t^{\text{targ}} = \hat{A}_t^{\text{GAE}} + V(s_t)$$

### Bias-Variance Properties of GAE
- $\lambda = 0$: $\hat{A}_t = \delta_t^V$. **Minimum variance, maximum bias** (pure one-step TD).
- $\lambda = 1$: $\hat{A}_t = \sum_{l=0}^\infty \gamma^l r_{t+l} - V(s_t) = G_t - V(s_t)$. **Maximum variance, zero bias** (pure Monte Carlo).
- In practice, $\lambda = 0.95$ provides the optimal trade-off, cutting gradient variance by over $80\%$ while maintaining low bias.

---

## 6.6 The Full Composite PPO Loss Function

In deep neural networks with shared representations, PPO optimizes a unified multi-task objective:
$$\mathcal{L}^{\text{PPO}}(\theta) = \hat{\mathbb{E}}_t \left[ L^{\text{CLIP}}(\theta) - c_1 L^{\text{VF}}(\theta) + c_2 S[\pi_\theta](s_t) \right]$$

### 1. Clipped Value Function Loss ($L^{\text{VF}}$)
To prevent the critic network from making destructive updates during multi-epoch optimization:
$$L^{\text{VF}}(\theta) = \frac{1}{2} \hat{\mathbb{E}}_t \left[ \max\left( (V_\theta(s_t) - \hat{R}_t^{\text{targ}})^2, \; (V_{\text{clip}}(s_t) - \hat{R}_t^{\text{targ}})^2 \right) \right]$$
Where $V_{\text{clip}}(s_t) = V_{\theta_{\text{old}}}(s_t) + \text{clip}\left( V_\theta(s_t) - V_{\theta_{\text{old}}}(s_t), -\epsilon_v, \epsilon_v \right)$.

### 2. Policy Entropy Bonus ($S[\pi_\theta]$)
Encourages exploration by penalizing premature certainty in action selection:
$$S[\pi_\theta](s_t) = - \sum_{a \in \mathcal{A}} \pi_\theta(a \mid s_t) \log \pi_\theta(a \mid s_t)$$

### Canonical Hyperparameter Specifications

| Hyperparameter | Symbol | Standard Value (Discrete / Atari) | Standard Value (Continuous / MuJoCo) |
| :--- | :---: | :---: | :---: |
| Horizon / Rollout Steps | $T_{\text{rollout}}$ | $128$ steps $\times$ 8 envs = 1024 | $2048$ steps $\times$ 1 env = 2048 |
| Optimizer / Learning Rate | Adam / $\alpha$ | $2.5 \times 10^{-4}$ (linearly annealed) | $3.0 \times 10^{-4}$ (linearly annealed) |
| Discount Factor | $\gamma$ | $0.99$ | $0.99$ |
| GAE Parameter | $\lambda$ | $0.95$ | $0.95$ |
| Clipping Range | $\epsilon$ | $0.1 - 0.2$ | $0.2$ |
| Value Loss Coefficient | $c_1$ | $0.5$ | $0.5$ |
| Entropy Coefficient | $c_2$ | $0.01$ | $0.0$ |
| PPO Epochs per Rollout | $K$ | $4 - 10$ epochs | $10$ epochs |
| Mini-batch Size | $B$ | $32 - 64$ transitions | $64$ transitions |
| Gradient Clipping Norm | $\max\lVert g \rVert$ | $0.5$ | $0.5$ |

---

## 6.7 Complete PPO Training Algorithm & Engineering Implementations

```
Algorithm: Proximal Policy Optimization (PPO-Clip with Vectorized Environs)
========================================================================================
1: Initialize policy/value network parameters theta
2: Initialize N_envs parallel environment instances
3: for iteration = 1 to Total_Iterations do
4:     // Stage 1: Vectorized Rollout Collection
5:     for step = 1 to T_rollout do
6:         Run policy pi_theta(s_t) across all N_envs
7:         Sample actions a_t, query values V(s_t), record log pi(a_t | s_t)
8:         Step environments: obtain r_t, s_{t+1}, done_t
9:         Append transitions to local rollout storage
10:    end for
11:    // Stage 2: GAE Advantage Calculation
12:    Evaluate terminal values V(s_{T+1})
13:    for t = T_rollout down to 1 do
14:        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
15:        A_t^GAE = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}^GAE
16:        R_t^targ = A_t^GAE + V(s_t)
17:    end for
18:    Flatten rollout buffer into dataset of size (N_envs * T_rollout)
19:    Normalize advantages: A_t <- (A_t - mean(A)) / (std(A) + 1e-8)
20:    // Stage 3: Multi-Epoch Mini-Batch SGD
21:    for epoch = 1 to K do
22:        Shuffle dataset into random mini-batches of size B
23:        for mini_batch in dataset do
24:            Evaluate pi_theta(a | s), V_theta(s), and Entropy S
25:            r(theta) = exp(log pi_theta(a | s) - log pi_old(a | s))
26:            L_CLIP = min(r * A, clip(r, 1-eps, 1+eps) * A)
27:            L_VF = (V_theta(s) - R^targ)^2
28:            L_total = - L_CLIP + c_1 * L_VF - c_2 * S
29:            Zero gradients, backpropagate L_total
30:            Clip gradient norms: clip_grad_norm_(theta, max_grad_norm=0.5)
31:            Optimizer step (Adam)
32:        end for
33:    end for
34:    Anneal learning rate linearly: alpha <- alpha_0 * (1 - iter / Total_Iterations)
35: end for
========================================================================================
```

---

## 6.8 Complex Locomotion: BipedalWalker-v3 PPO Benchmark

In `BipedalWalker-v3`, PPO demonstrates the capability to solve continuous locomotion with 24 state dimensions and 4 continuous torque outputs.

### Diagnostic Metric Dynamics on BipedalWalker
1. **Episode Return**: Traverses three regimes:
   - Early ($0 - 100\text{k}$ steps): Return $\sim -35.8$. Lander learns posture preservation and upright balance.
   - Mid ($500\text{k}$ steps): Return $\sim +109.3$. Agent discovers alternating limb strides.
   - Late ($2\text{M}$ steps): Return $\sim +295.1$. Fluid running gait achieved (solved at $\ge 300$).
2. **Policy Entropy**: Begins near $5.67$ ($H = \sum \log(\sqrt{2\pi e}\sigma)$ for 4 dimensions with $\sigma=1.0$). Steadily declines to $0.5 - 1.5$ as motor control narrows to precise gaits.
3. **Clip Fraction & Approximate KL**: Clip fraction remains between $0.08$ and $0.18$. Approximate KL divergence hovers safely around $0.015$, confirming stability within the trust region.

---

## 6.9 Comprehensive Game Project Evaluation Benchmark

The following cross-game benchmark synthesizes PPO performance across 14 gaming domains:

| Environment | Observation Type | Action Space | Horizon Difficulty | Exploration Difficulty | PPO Applicability / Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Flappy Bird** | Low-dim vector | Discrete (2) | Short | Low | **Optimal**: Solved in $< 100\text{k}$ steps. Minimal closed loop. |
| **Snake** | Grid / Pixel | Discrete (4) | Medium | Low | **High**: Solves food navigation; susceptible to self-trapping loops. |
| **2048** | Board state (16) | Discrete (4) | Medium | High | **Medium**: Prone to greedy merges; lacks long-range planning. |
| **Tetris** | Board state (200) | Discrete (sparse) | Long | Medium | **Challenging**: Requires terminal clearing foresight; easily tops out. |
| **Breakout** | Pixels (84x84) | Discrete (4) | Long | Medium | **High**: Solves tunneling exploit reliably with frame stacking. |
| **Procgen** | Procedural pixels | Discrete (15) | Medium | Medium | **Benchmark**: Evaluates generalization vs. visual memorization. |
| **CarRacing** | Pixels (96x96) | Continuous (3) | Long | Low | **High**: Successfully learns continuous steering, throttle, braking. |
| **Super Mario** | Pixels (240x256) | Multi-discrete | Long | High | **Medium**: Reward shaping dictates ceiling; gets stuck on pits/pipes. |
| **Sonic (Retro)** | Pixels | Multi-discrete | Long | High | **Challenging**: Speed causes visual outrunning; benefits from transfer. |
| **SoccerTwos** | Low-dim vector | Multi-discrete | Medium | High | **High**: Requires multi-agent self-play curriculum. |
| **ViZDoom** | First-person pixels| Discrete | Long | High | **High**: Solves deathmatch; requires recurrent LSTM for POMDP. |
| **Pokémon Red** | GameBoy pixels | Discrete (8) | Extreme ($> 10^5$) | Extreme | **Fails without guidance**: Extreme horizon credit assignment bottleneck. |
| **Crafter** | Top-down pixels | Discrete (17) | Long ($> 10^4$) | High | **Bottleneck**: Technology tree crafting requires intrinsic curiosity. |
| **NetHack** | ASCII grid / text | Discrete (large) | Extreme | Extreme | **Upper Limit**: Severe credit assignment breakdown; requires symbolic priors. |

---

## 6.10 Long-Horizon Tasks, Exploration Frontiers, & Planning Paradigms

When environments feature delayed or sparse rewards spanning thousands of timesteps, standard model-free PPO experiences credit assignment failure. Advanced frameworks augment policy gradients:

### 1. Hierarchical Reinforcement Learning (HRL) & Options
- **Options Framework (Sutton, Precup, Singh, 1999)**: Formulates an option $\omega \in \Omega$ as a 3-tuple $\langle \mathcal{I}_\omega, \pi_\omega, \beta_\omega \rangle$:
  1. Initiation set $\mathcal{I}_\omega \subseteq \mathcal{S}$: States where option can begin.
  2. Intra-option policy $\pi_\omega(a \mid s)$: High-frequency primitive control.
  3. Termination condition $\beta_\omega(s) \in [0, 1]$: Probability of option ending in state $s$.
- A higher-level policy-over-options $\pi_{\Omega}(\omega \mid s)$ selects options at low temporal frequency, reducing effective trajectory horizon from $10,000$ primitive steps to $50$ option steps.

### 2. Goal-Conditioned RL & Hindsight Experience Replay (HER)
In goal-conditioned RL, policies take states and desired goals: $\pi(a \mid s, g)$ with sparse binary reward $r(s, a, g) = -\mathbf{1}[s \neq g]$.
- **HER (Andrychowicz et al., 2017)**: In sparse-reward robotic manipulation (e.g., reaching, pushing), an agent fails in $99.9\%$ of initial episodes, receiving only $-1$ penalties. HER inspects a failed trajectory $(s_0, s_1, \ldots, s_T)$ and creates virtual replay transitions by replacing the original goal $g$ with the state actually reached: $g' = s_T$.
- *Insight*: Even if the agent failed to reach the intended target, it successfully demonstrated how to reach state $s_T$. HER converts every failure into a successful demonstration of an alternate goal, eliminating the sparse reward exploration barrier.

### 3. Imitation Learning from Demonstrations
When environment exploration is intractable, expert trajectories $\mathcal{D}_{\text{demo}} = \{\tau_1, \ldots, \tau_M\}$ bootstrap learning:
1. **Behavioral Cloning (BC)**: Supervised learning minimizing Cross-Entropy / MSE:
   $$\mathcal{L}_{\text{BC}}(\theta) = - \mathbb{E}_{(s, a) \sim \mathcal{D}_{\text{demo}}} \left[ \log \pi_\theta(a \mid s) \right]$$
   *Failure Mode (Covariate Shift)*: Small errors compound over time, drifting the agent into states unvisited by the expert where the policy exhibits undefined behavior.
2. **DAgger (Dataset Aggregation - Ross et al., 2011)**: Iterative interactive expert querying:
   - Roll out current policy $\pi_\theta$ to collect new state trajectory $\mathcal{D}_{\pi}$.
   - Query human/oracle expert to label the correct action $a^* = \pi^*(s)$ for every visited state in $\mathcal{D}_{\pi}$.
   - Aggregate $\mathcal{D} \leftarrow \mathcal{D} \cup \mathcal{D}_{\pi}$ and retrain $\pi_\theta$. Provably bounds compounding error to $\mathcal{O}(T)$.
3. **Generative Adversarial Imitation Learning (GAIL - Ho & Ermon, 2016)**: Formulates imitation as a minimax game: a Discriminator $D_\psi(s, a)$ distinguishes expert transitions from policy transitions, while policy $\pi_\theta$ optimizes PPO using discriminator confidence as reward:
   $$r(s, a) = - \log(1 - D_\psi(s, a))$$


# 7. Chapter 11: Continuous Control & Advanced Architectures

### Sources
- `docs/en/chapter11_continuous_control/ddpg.md`
- `docs/en/chapter11_continuous_control/td3-sac.md`
- `docs/en/chapter11_continuous_control/model-based.md`
- `docs/en/chapter11_continuous_control/search-world-models.md`

---

## 7.1 Deterministic Policy Gradients (DPG) & Deep DPG (DDPG)

In continuous action domains ($\mathcal{A} \subseteq \mathbb{R}^d$), traditional Q-learning fails because evaluating $\arg\max_a Q(s, a)$ requires solving an internal non-linear optimization problem at every control step. DPG (Silver et al., 2014) and DDPG (Lillicrap et al., 2015) resolve this by parameterizing a **deterministic actor** $\mu_\theta: \mathcal{S} \to \mathcal{A}$.

### The Deterministic Policy Gradient Theorem
Let $J(\mu_\theta) = \mathbb{E}_{s \sim \rho^\mu}[r(s, \mu_\theta(s))]$ be the continuous performance objective. The gradient with respect to actor weights $\theta$ is:
$$\nabla_\theta J(\mu_\theta) = \int_{\mathcal{S}} \rho^\mu(s) \nabla_\theta \mu_\theta(s) \left. \nabla_a Q^\mu(s, a) \right|_{a = \mu_\theta(s)} ds$$
In expectation over state visitation distribution $\rho^\mu$:
$$\nabla_\theta J(\mu_\theta) = \mathbb{E}_{s \sim \rho^\mu} \left[ \nabla_\theta \mu_\theta(s) \left. \nabla_a Q^\mu(s, a) \right|_{a = \mu_\theta(s)} \right]$$

### The Vector Chain Rule Interpretation
Applying the multivariate chain rule:
$$\nabla_\theta Q(s, \mu_\theta(s)) = \left( \nabla_\theta \mu_\theta(s) \right)^T \left. \nabla_a Q(s, a) \right|_{a = \mu_\theta(s)}$$
- $\nabla_a Q(s, a) \in \mathbb{R}^d$: The Critic indicates which direction in continuous action space produces the steepest increase in expected return.
- $\nabla_\theta \mu_\theta(s) \in \mathbb{R}^{d \times |\theta|}$: The Actor Jacobian indicates how adjusting network parameters shifts the output action vector.
- The Actor modifies its parameters so its output action moves in the direction recommended by the Critic.

### Off-Policy Deterministic Actor-Critic (DDPG)
Crucially, the DPG theorem holds off-policy: states can be sampled from an arbitrary behavioral replay distribution $\rho^\beta$:
$$\nabla_\theta J_\beta(\mu_\theta) \approx \mathbb{E}_{s \sim \rho^\beta} \left[ \nabla_\theta \mu_\theta(s) \left. \nabla_a Q^\mu(s, a) \right|_{a = \mu_\theta(s)} \right]$$
This permits full integration with an Experience Replay Buffer $\mathcal{D}$:
1. **Critic Optimization**: Minimizes Mean Squared Bellman Error:
   $$\mathcal{L}(\phi) = \frac{1}{|B|} \sum_{i \in B} \left( y_i - Q_\phi(s_i, a_i) \right)^2$$
   Where target is computed using target networks $\theta'$ and $\phi'$:
   $$y_i = r_i + \gamma (1 - d_i) Q_{\phi'}\left( s'_i, \mu_{\theta'}(s'_i) \right)$$
2. **Actor Optimization**: Maximizes predicted Q-value:
   $$\mathcal{L}(\theta) = - \frac{1}{|B|} \sum_{i \in B} Q_\phi\left( s_i, \mu_\theta(s_i) \right)$$
3. **Soft Polyak Target Synchronization**:
   $$\theta' \leftarrow \tau \theta + (1 - \tau) \theta', \quad \phi' \leftarrow \tau \phi + (1 - \tau) \phi' \quad (\tau \approx 0.005)$$
4. **Exploration Noise**: Because the policy is deterministic, exploration requires adding stochastic noise during interaction:
   $$a_t = \text{clip}\left( \mu_\theta(s_t) + \mathcal{N}(0, \sigma^2), a_{\text{low}}, a_{\text{high}} \right)$$
   (Or via Ornstein-Uhlenbeck process $dx_t = \theta_0(\mu_0 - x_t)dt + \sigma dW_t$ for temporally correlated physical inertia).

---

## 7.2 DDPG Failure Modes: Q-Overestimation & Brittleness

Despite theoretical elegance, basic DDPG suffers from three fatal instabilities in practice:
1. **Severe Q-Value Overestimation**:
   The target computation $y = r + \gamma Q_{\phi'}(s', \mu_{\theta'}(s'))$ uses the actor to choose actions that maximize $Q$. Due to approximation errors in function fitting, the critic overestimates values in certain regions. The actor exploits these overestimations, driving the system into regions of spurious high values that cause policy degeneration.
2. **Extreme Sensitivity to Hyperparameters**:
   A slight variation in learning rate, noise schedule, or reward scaling causes the policy to fall off performance cliffs from which it cannot recover.
3. **Exploitation of Sharp Q-Peaks**:
   The deterministic actor is susceptible to becoming trapped in sharp, narrow local maxima of the critic landscape that do not generalize across states.

---

## 7.3 Twin Delayed Deep Deterministic Policy Gradient (TD3)

TD3 (Fujimoto, van Hoof, & Meger, 2018) resolves all three core flaws of DDPG through three coordinated architectural modifications:

```
+----------------------------------------------------------------------------------------+
| 1. Clipped Double Q-Learning: Maintain Twin Critics Q_phi1, Q_phi2                     |
|    Target: y = r + gamma * min(Q_phi1'(s', a'), Q_phi2'(s', a'))                      |
|    --> Eliminates overestimation bias by taking the conservative lower bound           |
+----------------------------------------------------------------------------------------+
                                           |
                                           v
+----------------------------------------------------------------------------------------+
| 2. Target Policy Smoothing: Add clipped noise to target action                         |
|    a' = clip(mu_theta'(s') + epsilon, a_low, a_high), epsilon ~ clip(N(0, sigma^2),-c,c) |
|    --> Enforces smooth value manifold; prevents actor from chasing sharp Q-spikes      |
+----------------------------------------------------------------------------------------+
                                           |
                                           v
+----------------------------------------------------------------------------------------+
| 3. Delayed Policy Updates: Update Actor & Targets every d steps (e.g. d = 2)           |
|    Critics update every step; Actor updates once every 2 critic updates                |
|    --> Allows critics to stabilize before the actor takes gradient steps               |
+----------------------------------------------------------------------------------------+
```

### 1. Clipped Double Q-Learning
TD3 maintains two independent critic networks $Q_{\phi_1}, Q_{\phi_2}$ and two target critics $Q_{\phi_1'}, Q_{\phi_2'}$. Both critics are trained toward a shared conservative target:
$$y = r + \gamma (1 - d) \min_{i=1, 2} Q_{\phi_i'}\left( s', \tilde{a}' \right)$$
Taking the minimum suppresses value overestimation. Even if one critic overestimates the value of action $\tilde{a}'$, the minimum selects the less optimistic estimate, preventing overestimation from propagating backward.

### 2. Target Policy Smoothing
When updating the critic, value estimates can exploit narrow peaks in $Q_\phi$. TD3 smooths the target action by adding clipped Gaussian noise:
$$\tilde{a}' = \text{clip}\left( \mu_{\theta'}(s') + \epsilon, a_{\text{low}}, a_{\text{high}} \right), \quad \epsilon \sim \text{clip}\left( \mathcal{N}(0, \tilde{\sigma}^2), -c, c \right)$$
(Typically $\tilde{\sigma} = 0.2, c = 0.5$).
Smoothing acts as a regularizer: it forces neighboring actions to share similar value expectations, ensuring that the critic evaluates a robust neighborhood rather than an isolated point.

### 3. Delayed Policy Updates
Actor updates are decoupled from critic updates:
- Critics $\phi_1, \phi_2$ update on every mini-batch ($1:1$ ratio).
- Actor $\theta$ and target networks $\theta', \phi_1', \phi_2'$ update once every $d$ critic updates (standard setting $d = 2$).
This guarantees that the value function has settled into a lower-variance approximation before the policy takes a gradient step, eliminating policy chasing of unstable critic values.

---

## 7.4 Soft Actor-Critic (SAC): Maximum Entropy Reinforcement Learning

Soft Actor-Critic (Haarnoja et al., 2018) reformulates reinforcement learning under the **Maximum Entropy RL** framework:

### The Maximum Entropy Objective
The agent optimizes a dual objective: maximize expected return while maximizing policy entropy at every visited state:
$$J(\pi) = \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^{\infty} \gamma^t \left( r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot \mid s_t)) \right) \right]$$
Where policy entropy is:
$$\mathcal{H}(\pi(\cdot \mid s_t)) = \mathbb{E}_{a \sim \pi} \left[ - \log \pi(a \mid s_t) \right]$$
The temperature parameter $\alpha > 0$ balances exploitation (accumulating reward) against exploration (maintaining randomness).

### Soft Bellman Equation & Twin Critic Targets
The soft action-value function satisfies:
$$Q(s, a) = r + \gamma \mathbb{E}_{s' \sim P}[V(s')]$$
Where the soft state-value is defined as:
$$V(s) = \mathbb{E}_{a \sim \pi} \left[ Q(s, a) - \alpha \log \pi(a \mid s) \right]$$
Combining with twin critics ($Q_{\phi_1}, Q_{\phi_2}$) to eliminate overestimation bias, the target regression value is:
$$y = r + \gamma (1 - d) \left( \min_{j=1, 2} Q_{\phi_j'}(s', \tilde{a}') - \alpha \log \pi_\theta(\tilde{a}' \mid s') \right), \quad \tilde{a}' \sim \pi_\theta(\cdot \mid s')$$

### Reparameterization Trick with Tanh Squashing
The actor parameterizes a Gaussian distribution. To sample actions differentiably while bounding actions within $[-1, 1]$:
1. Sample standard normal noise: $\epsilon \sim \mathcal{N}(0, I)$.
2. Compute unbounded latent action via reparameterization:
   $$u = \mu_\theta(s) + \sigma_\theta(s) \odot \epsilon$$
3. Squash through hyperbolic tangent:
   $$a = \tanh(u) \in (-1, 1)$$

#### Exact Jacobian Log-Probability Correction
Because the non-linear squashing transformation $a = \tanh(u)$ distorts probability density, the policy log-probability must be corrected using the Jacobian determinant:
$$\pi(a \mid s) = \mu(u \mid s) \left| \det\left( \frac{da}{du} \right) \right|^{-1}$$
Since $\frac{da_i}{du_i} = 1 - \tanh^2(u_i)$, the log-probability is:
$$\log \pi_\theta(a \mid s) = \log \mu_\theta(u \mid s) - \sum_{i=1}^d \log\left( 1 - \tanh^2(u_i) + \epsilon_{\text{num}} \right)$$
Where $\log \mu_\theta(u \mid s) = - \frac{1}{2} \sum_{i=1}^d \left[ \left(\frac{u_i - \mu_i(s)}{\sigma_i(s)}\right)^2 + 2\log\sigma_i(s) + \log(2\pi) \right]$ and $\epsilon_{\text{num}} = 10^{-6}$ prevents numerical underflow.

### Actor Objective in SAC
The actor minimizes the KL divergence between policy and the softmax distribution of Q-values:
$$\mathcal{L}_{\text{actor}}(\theta) = \mathbb{E}_{s \sim \mathcal{D}, \epsilon \sim \mathcal{N}} \left[ \alpha \log \pi_\theta(\tilde{a}_\theta(s, \epsilon) \mid s) - \min_{j=1, 2} Q_{\phi_j}(s, \tilde{a}_\theta(s, \epsilon)) \right]$$
Gradients backpropagate through both the policy entropy term and the critic network via the sampled $\tilde{a}_\theta(s, \epsilon)$.

### Automatic Temperature Tuning
Manually tuning $\alpha$ across different environments is difficult because reward scales vary. SAC formulates temperature optimization as a dual constrained optimization problem: maximize return subject to an average entropy constraint $\mathcal{H}(\pi) \ge \mathcal{H}_0$:
$$\mathcal{H}_0 = - \dim(\mathcal{A}) \quad \text{(heuristic target entropy)}$$
The dual objective for temperature parameter $\alpha$ is:
$$\mathcal{L}(\alpha) = \mathbb{E}_{s \sim \mathcal{D}, a \sim \pi} \left[ - \alpha \left( \log \pi_\theta(a \mid s) + \mathcal{H}_0 \right) \right]$$
- If policy entropy drops below $\mathcal{H}_0$ ($\log \pi > -\mathcal{H}_0$), $\alpha$ increases, boosting exploration.
- If policy entropy exceeds $\mathcal{H}_0$, $\alpha$ decreases, focusing on exploitation.

---

## 7.5 Comparison: DDPG vs. TD3 vs. SAC

| Architectural Feature | DDPG (2015) | TD3 (2018) | SAC (2018) |
| :--- | :--- | :--- | :--- |
| **Policy Type** | Deterministic: $a = \mu(s)$ | Deterministic: $a = \mu(s)$ | **Stochastic**: $a \sim \pi(\cdot \mid s)$ |
| **Action Generation** | $\mu(s) + \text{noise}$ | $\mu(s) + \text{noise}$ | **Reparameterized Tanh Gaussian** |
| **Objective Formulation** | Standard Expected Return | Standard Expected Return | **Maximum Entropy RL** |
| **Critic Networks** | 1 Critic ($Q_\phi$) | **Twin Critics** ($Q_{\phi_1}, Q_{\phi_2}$) | **Twin Critics** ($Q_{\phi_1}, Q_{\phi_2}$) |
| **Target Network Updates**| Soft Polyak ($\tau$) | **Delayed Soft Polyak** (every $d=2$) | Soft Polyak ($\tau$) |
| **Target Action Regularization**| None | **Target Policy Smoothing** | **Entropy Penalty** ($\alpha \log \pi$) |
| **Exploration Strategy** | Heuristic additive noise | Heuristic additive noise | **Learned maximum entropy** |
| **Temperature Parameter** | None | None | **Automatically tuned** $\alpha$ |
| **Overestimation Handling**| None (Severe bias) | **Clipped Double Q** ($\min Q$) | **Clipped Double Q** ($\min Q$) |
| **Benchmark Performance** | Prone to sudden divergence | State-of-the-art on MuJoCo | **Highest sample efficiency & stability** |

---

## 7.6 Model-Based Reinforcement Learning: Paradigms & Compounding Error

Model-free algorithms (PPO, TD3, SAC) require $10^6 - 10^7$ environment interactions because they learn strictly by trial and error. **Model-Based RL** learns explicit predictive models of environment dynamics:
$$\hat{P}(s_{t+1} \mid s_t, a_t) \quad \text{and} \quad \hat{R}(s_t, a_t)$$

### Compounding Error: The Fundamental Hurdle of Model-Based RL
Let single-step model prediction error be bounded by $\epsilon$:
$$\lVert \hat{s}_{t+1} - s_{t+1}^* \rVert \le \epsilon$$
Assuming Lipschitz continuous dynamics with constant $L > 1$:
$$\lVert \hat{s}_{t+1} - s_{t+1}^* \rVert \le L \lVert \hat{s}_t - s_t^* \rVert + \epsilon$$
Unrolling over a planning horizon of $T$ timesteps:
$$e_T \le \epsilon \sum_{i=0}^{T-1} L^i = \epsilon \frac{L^T - 1}{L - 1} = \mathcal{O}(L^T \epsilon)$$
*Takeaway*: Model prediction errors compound **exponentially** with rollout horizon $T$. A policy optimized inside an inaccurate long-horizon simulation exploits the simulation's flaws (model exploitation), achieving high imagined returns while failing completely in the real environment.

---

## 7.7 Dyna-Q, PETS, and Model-Based Policy Optimization (MBPO)

### 1. Dyna-Q (Model as Data Generator)
Interleaves real-world interactions with simulated planning steps:
- Execute real action, observe $(s, a, r, s')$, update model $\hat{P}, \hat{R}$.
- Perform standard Q-learning step on real transition.
- **Dyna Planning**: Repeat $K$ times: sample random previously observed state $s$, sample past action $a$, query model for predicted $(r, s')$, and perform Q-learning update on simulated transition.

### 2. PETS: Probabilistic Ensembles with Trajectory Sampling (Chua et al., 2018)
PETS explicitly decouples and quantifies two forms of uncertainty:
- **Aleatoric Uncertainty**: Inherent environmental randomness (modeled by parameterizing Gaussian output distributions $\mathcal{N}(\mu(s, a), \Sigma(s, a))$).
- **Epistemic Uncertainty**: Lack of training data (modeled by training an **ensemble** of $B$ bootstrap models).

#### Planning via Model Predictive Control (MPC) and Cross-Entropy Method (CEM)
PETS does not train a neural policy. Instead, it plans online at every step using CEM:
1. Initialize action sequence distribution $\mathcal{N}(\mu_t, \Sigma_t)$ for horizon $H$.
2. Sample $P$ candidate action sequences.
3. Propagate state trajectories through ensemble particles via trajectory sampling.
4. Select top $K$ elite action sequences yielding highest cumulative return.
5. Refit $\mu_t, \Sigma_t$ to elite sequences; iterate.
6. Execute only the first action $a_0^*$; discard the rest and replan at the next timestep (receding horizon).

### 3. Model-Based Policy Optimization (MBPO - Janner et al., 2019)
MBPO resolves compounding error by restricting model rollouts to **extremely short branched horizons** ($k = 1$ to $15$ steps):
- Maintains an ensemble of dynamics models trained on real replay buffer $\mathcal{D}_{\text{env}}$.
- Starts model rollouts **exclusively from real states** sampled from $\mathcal{D}_{\text{env}}$:
  $$s_0 \sim \mathcal{D}_{\text{env}}$$
- Roll out current policy $\pi_\theta$ using model ensemble for only $k$ steps, generating short synthetic transitions stored in model buffer $\mathcal{D}_{\text{model}}$.
- Train an off-policy model-free agent (SAC) on a mixture:
  $$\mathcal{B} \sim (1 - \beta) \mathcal{D}_{\text{env}} + \beta \mathcal{D}_{\text{model}}$$
*Theoretical Guarantee*: MBPO proves that if rollout length $k \le \mathcal{O}(\epsilon^{-1})$, the return of the policy under the model bounds the return in the true environment monotonically.

---

## 7.8 Search & World Models: AlphaZero, MuZero, and Dreamer V3

The frontier of reinforcement learning merges deep representation learning with forward search and latent world models:

### 1. AlphaZero (Silver et al., 2018)
Unifies deep neural networks with Monte Carlo Tree Search (MCTS) under perfect simulator dynamics (chess, shogi, Go):
- Node selection uses the Predictor Upper Confidence Tree (PUCT) formula:
  $$\text{PUCT}(s, a) = Q(s, a) + c_{\text{prior}} P(s, a) \frac{\sqrt{N(s)}}{1 + N(s, a)}$$
- During self-play, MCTS acts as a powerful policy improvement operator. The neural network $\pi_\theta, v_\theta$ is continuously trained to predict the improved MCTS visit distribution $\boldsymbol{\pi}_{\text{MCTS}}$ and the game outcome $z$.

### 2. MuZero: Planning Without Environment Models (Schrittwieser et al., Nature 2020)
AlphaZero requires a perfect simulator to evaluate legal moves and state transitions. MuZero removes this requirement by learning an **implicit latent model**:

```
Observation o_t
       |
       v
[Representation Network h_theta] ---> Latent State s_0
                                            |
                                            +--------------------+
                                            |                    |
                                            v                    v
                                     [Prediction f_theta] [Dynamics g_theta]
                                            |                    |
                                            v                    v
                                     Policy p_0, Value v_0  Reward r_1, Latent s_1
                                                                 |
                                                                 +---> [Prediction f_theta]
                                                                 |           |
                                                                 v           v
                                                        [Dynamics g_theta] Policy p_1, Value v_1
                                                                 |
                                                                 v
                                                            Reward r_2, Latent s_2 ...
```

#### The Three Core Networks of MuZero
1. **Representation Network ($h_\theta$)**: Maps past observation history to an abstract latent state:
   $$s_0 = h_\theta(o_1, \ldots, o_t)$$
2. **Dynamics Network ($g_\theta$)**: Predicts next latent state and immediate reward given latent state and candidate action:
   $$r_k, s_k = g_\theta(s_{k-1}, a_k)$$
3. **Prediction Network ($f_\theta$)**: Predicts policy distribution and state-value directly from latent state:
   $$p_k, v_k = f_\theta(s_k)$$

*Significance*: MuZero does not reconstruct raw observation pixels (which waste capacity modeling task-irrelevant backgrounds). It learns latent representations predictive **only of quantities relevant to planning**: rewards, policy priors, and values.

### 3. Dreamer V3: Mastering Diverse Domains in Latent Imagination (Hafner et al., 2023)
Dreamer V3 trains an Actor-Critic agent entirely inside the "dreams" (imagined rollouts) of a learned **Recurrent State-Space Model (RSSM)**:
- **Discrete Autoencoders**: Represents latent states using categorical distributions, eliminating representation collapse.
- **Actor-Critic in Imagination**: The agent rolls out imagined trajectories $15$ steps into the future within the RSSM latent space, optimizing policy and value heads without touching the real environment.
- **Symlog Transformation**: Transforms inputs and targets via $\text{symlog}(x) = \text{sign}(x) \ln(|x| + 1)$ and symmetric exponential $\text{symexp}(x)$, stabilizing optimization across wildly divergent reward scales (from Atari scores of $1.0$ to Minecraft scores of $10,000$).
- *Achievement*: The first algorithm to collect diamonds in Minecraft from scratch without human demonstrations or domain-specific crafting heuristics.

---

## Complete Part 1 Methodological Synthesis

```
+--------------------------------------------------------------------------------------------------+
|                                  CLASSICAL & DEEP RL SPECTRUM                                    |
+--------------------------------------------------------------------------------------------------+
| Paradigm             | Key Algorithms     | Core Equations / Target Mechanics   | Primary Limitation     |
+----------------------+--------------------+-------------------------------------+------------------------+
| Tabular Exact        | Value Iteration,   | V = max [R + gamma * sum P * V]     | Intractable for large  |
|                      | Policy Iteration   | v_pi = (I - gamma * P_pi)^-1 * r    | or continuous states   |
+----------------------+--------------------+-------------------------------------+------------------------+
| Tabular Model-Free   | Q-Learning,        | Q <- Q + alpha[r + gamma*max Q - Q] | Zero generalization;   |
|                      | SARSA              | delta = r + gamma * Q' - Q          | discrete tabular only  |
+----------------------+--------------------+-------------------------------------+------------------------+
| Deep Q-Learning      | DQN, Double DQN,   | L = Huber(r + gamma*Q(s',a*;theta-) | Moving targets;        |
|                      | Dueling, Rainbow   |           - Q(s,a;theta))           | discrete actions only  |
+----------------------+--------------------+-------------------------------------+------------------------+
| Policy Gradient      | REINFORCE          | grad J = E[sum grad log pi * G_t]   | High variance;         |
|                      |                    | L = - sum log pi * (G_t - V(s))     | episodic rollout only  |
+----------------------+--------------------+-------------------------------------+------------------------+
| Actor-Critic         | A2C, V-trace,      | delta = r + gamma * V(s') - V(s)    | Bias from critic;      |
|                      | AlphaStar          | grad J = E[grad log pi * delta]     | gradient interference  |
+----------------------+--------------------+-------------------------------------+------------------------+
| Trust Region PPO     | PPO-Clip, GAE      | L = min(r*A, clip(r,1+-eps)*A)      | On-policy sample       |
|                      |                    | A^GAE = sum (gamma*lambda)^l delta  | efficiency bottleneck  |
+----------------------+--------------------+-------------------------------------+------------------------+
| Continuous Control   | DDPG, TD3,         | grad J = E[grad_theta mu * grad_a Q]| Sensitive to value     |
|                      | SAC                | Target: min(Q1, Q2) - alpha*log pi  | manifold exploitation  |
+----------------------+--------------------+-------------------------------------+------------------------+
| Model-Based & World  | PETS, MBPO,        | e_T <= O(L^T * eps)                 | Compounding error;     |
| Models               | MuZero, Dreamer V3 | RSSM Latent Imagination Rollouts    | model capacity limits  |
+----------------------+--------------------+-------------------------------------+------------------------+
```


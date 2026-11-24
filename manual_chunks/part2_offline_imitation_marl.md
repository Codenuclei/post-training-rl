# Comprehensive Technical Manual: Offline RL, Imitation Learning, Meta-RL, Exploration, MARL, and Hierarchical Decision-Making

This manual presents an exhaustive technical extraction of the foundations, algorithms, mathematical formulations, hyperparameter configurations, failure modes, code implementations, and comparative analyses from:
1. `chapter12_offline_rl` (Sections 10.1, 10.2, 10.3)
2. `chapter13_imitation_meta_rl` (Sections 11.1, 11.2, 11.3)
3. `chapter14_exploration_marl_hierarchical` (Sections 12.1, 12.2, 12.3)

---

# Part I: Offline Reinforcement Learning (Chapter 12)

## 1. Offline Data and Distribution Shift (Section 10.1)

### 1.1 Problem Formulation & Why Fixed Data Cause Distribution Shift
In online reinforcement learning (e.g., DQN, SAC, PPO), policy exploration provides corrective feedback: if the value function temporarily overestimates an action or state, the agent visits it, receives true environment feedback, and corrects the estimate. The standard one-step temporal-difference target is:

$$y = r + \gamma \cdot \mathbb{E}_{s' \sim P(\cdot \mid s, a)}\left[V(s')\right]$$

In offline RL, the dataset $\mathcal{D}$ is collected once by a behavior policy $\pi_\beta$ and remains **completely fixed**:

$$\mathcal{D} = \{(s_i, a_i, r_i, s'_i)\}_{i=1}^{N}, \quad (s, a) \sim d^{\pi_\beta}(s) \pi_\beta(a \mid s)$$

When a newly trained policy $\pi_\theta(a \mid s)$ departs from $\pi_\beta(a \mid s)$, it induces **distribution shift**.

### 1.2 Extrapolation Error Decomposition & Recursive Error Accumulation
Let the action support of the dataset be $\mathcal{D}_\mathcal{A}(s) = \{a : (s, a) \in \text{support}(\pi_\beta(\cdot \mid s))\}$. If the Bellman backup evaluates an out-of-distribution (OOD) action $a' \notin \mathcal{D}_\mathcal{A}(s')$, function approximators (neural networks) extrapolate arbitrarily.

#### Error Decomposition:
$$\underbrace{Q_\phi(s, a) - Q^\pi(s, a)}_{\text{total error}} = \underbrace{\epsilon_{\text{stat}}}_{\substack{\text{statistical error}\\\text{(finite samples)}}} + \underbrace{\epsilon_{\text{approx}}}_{\substack{\text{approximation error}\\\text{(network capacity)}}} + \underbrace{\max_{a'} Q_\phi(s', a') - Q^\pi(s', \pi(s'))}_{\text{extrapolation error}}$$

#### Error Accumulation Across Iterations:
Let $Q_0$ be the initial estimate. After $T$ Bellman iterations:

$$\|Q_T - Q^\pi\|_\infty \leq \gamma^T \|Q_0 - Q^\pi\|_\infty + \sum_{k=0}^{T-1} \gamma^k \|\mathcal{T} Q_k - \mathcal{T}^\pi Q_k\|_\infty$$

Where:
- $\mathcal{T}$ is the empirical Bellman update with action maximization.
- $\mathcal{T}^\pi$ is the update under the true policy.
- An OOD error $\epsilon_{\text{ood}}$ introduced at each step accumulates through a geometric series to approximately $\frac{\epsilon_{\text{ood}}}{1 - \gamma}$. For $\gamma = 0.99$, the amplification factor is $\approx 100\times$.

#### The Offline Optimization Objective:
$$\max_\theta \; \mathbb{E}_{s \sim \mathcal{D}}\left[Q^\pi(s, \pi_\theta(s))\right] \quad \text{subject to} \quad D(\pi_\theta \| \pi_\beta) \leq \epsilon$$

---

### 1.3 Conservative Value Algorithms: BCQ, CQL, and IQL

#### 1.3.1 BCQ (Batch-Constrained Q-Learning)
Constrains target action $a'$ to lie strictly within the support of $\pi_\beta$.
- Trains a conditional VAE to model $\pi_\beta(a \mid s)$.
- Generates candidate actions $\{a_i\} \sim \pi_\beta$ and perturbs them via perturbation network $\Phi(s, a_i)$ within range $[-\xi, \xi]$:

$$a' = \arg\max_{a \in \{a_i + \xi \Phi(s, a_i)\}} Q_\phi(s', a)$$

#### 1.3.2 CQL (Conservative Q-Learning)
Directly penalizes Q-values for OOD actions by augmenting standard Bellman error with a conservative regularizer:

$$\mathcal{L}_{\text{CQL}}(Q) = \alpha \left(\mathbb{E}_{s \sim \mathcal{D}}\left[\log \sum_a \exp(Q(s, a))\right] - \mathbb{E}_{(s, a) \sim \mathcal{D}}[Q(s, a)]\right) + \mathcal{L}_{\text{Bellman}}(Q)$$

- **Theoretical Lower Bound**: Learned $\hat{Q}$ satisfies $\hat{Q}(s, a) \le Q^\pi(s, a)$ pointwise, with OOD actions under-valued by an $\mathcal{O}(\alpha)$ gap.
- **Dual Lagrangian for Adaptive Conservatism**:
  $$\mathcal{L}(\alpha) = -\alpha \cdot \left(\mathbb{E}_s\left[\log\sum_a \exp(\hat{Q}(s, a))\right] - \mathbb{E}_{(s, a) \sim \mathcal{D}}[\hat{Q}(s, a)] - \xi\right)$$
  where $\xi$ is the target gap (typically $\xi = 5.0$).

```python
# CQL Critic Update PyTorch Implementation
class CQL(SAC):
    def critic_loss(self, batch):
        s, a, r, s_next, done = batch
        with torch.no_grad():
            a_next = self.actor(s_next)
            q_target = torch.min(self.critic_target1(s_next, a_next),
                                  self.critic_target2(s_next, a_next))
            y = r + self.gamma * (1 - done) * q_target
        bellman_loss = F.mse_loss(self.critic1(s, a), y) + F.mse_loss(self.critic2(s, a), y)

        # CQL regularizer: sample random (OOD) actions, evaluate current and next actions
        rand_a = torch.rand_like(a) * 2 - 1
        q_rand1 = self.critic1(s, rand_a).flatten()
        q_curr1 = self.critic1(s, a).flatten()  # in-distribution
        q_next1 = self.critic1(s, a_next).flatten()
        cat_q1 = torch.cat([q_rand1, q_curr1, q_next1], dim=1)
        logsumexp_q1 = torch.logsumexp(cat_q1, dim=1).mean()

        conservative_loss = (
            self.alpha * (logsumexp_q1 - q_curr1.mean())
            + self.alpha * (logsumexp_q2 - q_curr2.mean())
        )
        return bellman_loss + conservative_loss
```

#### 1.3.3 IQL (Implicit Q-Learning)
Completely avoids querying or maximizing over actions outside the dataset.
- Value function $V_\psi(s)$ fitted via **expectile regression**:
  $$\mathcal{L}_V = \mathbb{E}_{(s, a) \sim \mathcal{D}}\left[L_2^\tau(Q_{\bar{\theta}}(s, a) - V_\psi(s))\right]$$
  where the expectile loss is defined as:
  $$L_2^\tau(x) = |\tau - \mathbb{1}(x < 0)| \cdot x^2$$
  Setting $\tau \in (0.5, 1.0)$ (typically $\tau = 0.7$) biases $V$ toward the upper support of in-dataset Q-values.
- Advantage extraction: $A(s, a) = Q_{\bar{\theta}}(s, a) - V_\psi(s)$.
- Policy training via Advantage-Weighted Regression (AWR):
  $$\mathcal{L}_\pi = -\mathbb{E}_{(s, a) \sim \mathcal{D}}\left[\exp(\beta \cdot A(s, a)) \cdot \log \pi_\theta(a \mid s)\right]$$

#### Comparison of Value-Constrained Offline Algorithms:
| Dimension | BCQ | CQL | IQL |
| :--- | :--- | :--- | :--- |
| **Constraint location** | Action space | Value function | Implicit (expectile + AWR) |
| **Evaluates OOD actions** | No (sampling constraint) | Yes (logsumexp over random actions) | No (strictly queries in-dataset $(s, a)$) |
| **Additional network** | VAE $\pi_\beta$ + perturbation $\Phi$ | None | Separate $V(s)$ network |
| **Hyperparameter sensitivity**| High (perturbation magnitude $\xi$) | Medium (automatic $\alpha$ dual) | Low ($\tau \in [0.7, 0.9], \beta$) |
| **Medium dataset performance**| Medium | Strong | Strong |
| **Sparse dataset stability** | Medium | Occasionally unstable | Strong |
| **Implementation complexity** | High | Medium | Low |

---

### 1.4 Policy Regularization: TD3+BC and AWAC

#### 1.4.1 TD3+BC
Appends a behavior cloning penalty directly to the deterministic policy gradient:

$$\mathcal{L}_{\text{actor}} = -\mathbb{E}_{s \sim \mathcal{D}}\left[Q(s, \mu_\theta(s))\right] + \lambda \cdot \mathbb{E}_{(s, a) \sim \mathcal{D}}\left[(\mu_\theta(s) - a)^2\right]$$

Adaptive normalization factor:
$$\lambda = \frac{\alpha}{\frac{1}{N}\sum_i |Q(s_i, \mu_{\theta_{\text{old}}}(s_i))|}$$
- Standard setting: $\alpha = 2.5$ across all D4RL MuJoCo tasks.

#### 1.4.2 AWAC (Advantage-Weighted Actor-Critic)
$$\mathcal{L}_\pi^{\text{AWAC}} = -\mathbb{E}_{(s, a) \sim \mathcal{D}}\left[\exp\left(\frac{A(s, a)}{\beta}\right) \cdot \log \pi_\theta(a \mid s)\right]$$
- Advantage definition: $A(s, a) = Q_\phi(s, a) - V_\psi(s)$.
- Supports seamless offline pretraining to online fine-tuning.

#### Key Architectural Contrast: AWAC vs. IQL
- **AWAC**: $Q$ is trained using standard Bellman backups where the target evaluates $a' \sim \pi$.
- **IQL**: $Q$ is backed up through $V(s')$ without action maximization ($y = r + \gamma V(s')$), completely removing out-of-distribution evaluation.

| Method | Policy-Loss Form | Requires $V$ Network | Supports Online Fine-tuning |
| :--- | :--- | :--- | :--- |
| **TD3+BC** | $-Q + \lambda \|\mu - a\|^2$ | No | Medium |
| **AWAC** | $-w(A)\log \pi, \quad w=\exp(A/\beta)$ | Yes | Strong |
| **IQL** | $-\exp(\beta A)\log \pi$ (AWR) | Yes | Medium |

---

## 2. Offline RL via Sequence Modeling (Section 10.2)

### 2.1 Decision Transformer (DT)
Replaces dynamic programming / Bellman backups with conditional sequence prediction.

#### Return-to-Go (RTG):
$$\hat{R}_t = \sum_{t'=t}^{T} r_{t'}$$

#### Input Token Sequence:
Traversing causal context as interleaved triplets:
$$\hat{R}_1, s_1, a_1, \hat{R}_2, s_2, a_2, \ldots, \hat{R}_T, s_T, a_T$$

#### Causal Attention & Training Objective:
$$\pi_\theta(a_t \mid \hat{R}_t, s_t, a_{t-1}, \ldots) = \text{Transformer}(\hat{R}_{1:t}, s_{1:t}, a_{1:t-1})$$

$$\mathcal{L}_{\text{DT}} = \mathbb{E}_{\tau \sim \mathcal{D}}\left[\sum_t \|\hat{a}_t - a_t\|^2\right] \quad \text{(or Cross-Entropy for discrete actions)}$$

#### Theoretical Induced Policy:
$$\pi_\theta(a \mid s, \hat{R}) \propto \exp\left(-\frac{1}{2\sigma^2}\|a - f_\theta(s, \hat{R})\|^2\right) \approx \pi_\beta(a \mid s, \text{return} \approx \hat{R})$$

```python
# Decision Transformer Architecture & Autoregressive Act Step
class DecisionTransformer(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_dim, n_heads, n_layers, max_ep_len=4096):
        super().__init__()
        self.embed_rtg = nn.Linear(1, hidden_dim)
        self.embed_state = nn.Linear(state_dim, hidden_dim)
        self.embed_action = nn.Linear(act_dim, hidden_dim)
        self.embed_ln = nn.LayerNorm(hidden_dim)
        self.pos_emb = nn.Embedding(max_ep_len, hidden_dim)
        self.transformer = GPT(d_model=hidden_dim, n_heads=n_heads, n_layers=n_layers,
                               attn_pdrop=0.1, resid_pdrop=0.1)
        self.action_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, rtg, states, actions, timesteps):
        B, T, _ = states.shape
        rtg_emb = self.embed_rtg(rtg)
        state_emb = self.embed_state(states) + self.pos_emb(timesteps)
        action_emb = self.embed_action(actions)
        # Interleave tokens: (B, 3*T, H)
        stacked = torch.stack([rtg_emb, state_emb, action_emb], dim=1).permute(0, 2, 1, 3).reshape(B, 3 * T, -1)
        stacked = self.embed_ln(stacked)
        h = self.transformer(stacked)
        h_states = h[:, 1::3, :]  # State token positions predict next action
        return self.action_head(h_states)
```

#### Inference Execution:
Target RTG is initialized to an expert level (e.g., 9000 on HalfCheetah) and decremented step-by-step:
$$\hat{R}_{t+1} = \hat{R}_t - r_t$$

#### DT Limitations:
1. **Bounded by data coverage**: Cannot extrapolate beyond the best trajectory return in $\mathcal{D}$.
2. **Weak stitching**: Lacks dynamic programming's ability to compose suboptimal trajectory segments into a globally optimal path.
3. **Conditioning sensitivity**: Hallucinates or collapses if prompted with an out-of-distribution RTG target.

---

### 2.2 Trajectory Transformer (TT) & Diffuser

#### Trajectory Transformer (TT):
Discretizes all dimensions $(s, a, r)$ into uniform tokens and models the joint autoregressive probability:
$$p_\theta(\tau) = \prod_{t=1}^{T} p_\theta(s_t, a_t, r_t \mid s_{<t}, a_{<t}, r_{<t})$$
- Planning performed via **beam search** conditioned on cumulative reward constraints.

#### Diffuser:
Generates complete continuous trajectory matrices of shape $T \times (d_s + d_a)$ using a score-based diffusion model:
$$\min_\theta \; \mathbb{E}_{\tau, t, \epsilon}\left[\|\epsilon - \epsilon_\theta(\tau_t, t)\|^2\right]$$
Conditioning via **classifier-free guidance**:
$$\tilde{\epsilon}_\theta = (1 + w) \cdot \epsilon_\theta(\tau_t, t, c) - w \cdot \epsilon_\theta(\tau_t, t)$$
where $c$ is a conditioning variable (e.g., maximizing cumulative reward or achieving a terminal goal state).

#### Comparison of Sequence-Modeling Approaches:
| Dimension | Decision Transformer | Trajectory Transformer | Diffuser |
| :--- | :--- | :--- | :--- |
| **Modeling target** | Conditional policy given RTG | Joint distribution over trajectory tokens | Diffusion over continuous trajectory matrices |
| **Discretization** | None | Yes (each state & action dimension) | None |
| **Inference** | Autoregressive sampling | Beam search | Iterative denoising |
| **Planning ability** | Weak (implicit) | Strong (explicit token search) | Strong (conditional reward guidance) |
| **Stitching ability** | Weak | Medium | Strong |
| **Inference speed** | Fast | Slow | Medium (dozens of denoising steps) |
| **LLM stack compatibility**| Strong (standard GPT architecture)| Strong | Weak (custom spatial UNet / temporal convs) |

---

## 3. Offline RL and Preference Data (Section 10.3)

### 3.1 DPO (Direct Preference Optimization) as Implicit Q-Learning
The DPO objective derived from a KL-constrained offline RL problem:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$

Implicit advantage definition:
$$\hat{A}(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$

Token-level dynamic programming equivalence:
$$Q^*(s_t, a_t) = \hat{r}(s_t, a_t) + \gamma \mathbb{E}_{s_{t+1}}\left[\max_{a'} Q^*(s_{t+1}, a')\right]$$

### 3.2 Formal Mapping: D4RL vs. LLM Preference Optimization
| Dimension | D4RL (MuJoCo Offline Control) | LLM Preference Optimization |
| :--- | :--- | :--- |
| **State $s$** | Robot joint angles / velocities | Prompt context $x$ |
| **Action $a$** | Continuous joint torques | Response sequence $y$ / token $y_t$ |
| **Reward $r$** | Explicit scalar environment feedback | Implicit preference ranking $y_w \succ y_l$ |
| **Data source** | Behavior policy $\pi_\beta$ | Human annotators / reward model labels |
| **Objective** | $\max Q^\pi \quad \text{s.t.} \quad \pi \approx \pi_\beta$ | $\max \hat{r} \quad \text{s.t.} \quad \mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) \le \epsilon$ |
| **Algorithms** | CQL / IQL / DT | DPO / IPO / KTO / RLVR |

---

# Part II: Imitation Learning and Meta-RL (Chapter 13)

## 1. Behavior Cloning & Interactive Imitation Learning (Section 11.1)

### 1.1 Behavior Cloning (BC)
$$\mathcal{L}_{\text{BC}}(\theta) = -\mathbb{E}_{(s, a) \sim \mathcal{D}_{\text{expert}}}\left[\log \pi_\theta(a \mid s)\right]$$
Minimizes negative log-likelihood (or cross-entropy / MSE) under expert state distribution $d^{\pi^*}(s)$.

### 1.2 Compounding Error Analysis
Let the single-step error rate on expert-covered states be $\epsilon$. Across a horizon of $T$ steps:
- Probability of a zero-error trajectory: $(1 - \epsilon)^T$.
- Cumulative expected quadratic cost bound (Ross et al., 2011):

$$\mathbb{E}\left[\sum_{t=0}^T \mathbb{1}[\pi_\theta(s_t) \neq \pi^*(s_t)]\right] \leq \mathcal{O}(T^2 \epsilon)$$

| Task Horizon $T$ | 10 steps | 50 steps | 100 steps | 1,000 steps |
| :--- | :--- | :--- | :--- | :--- |
| **Zero-error probability ($\epsilon = 0.01$)** | 0.90 | 0.61 | 0.37 | $0.00004$ |

### 1.3 DAgger (Dataset Aggregation)
Eliminates covariate shift by training on states visited by the agent's own policy:
1. Roll out current policy $\pi_{\theta}$ mixed with expert: $\beta = \max(0.0, 1.0 - \text{iter}/10)$.
2. Collect visited states: $\mathcal{S}_{\text{visited}} = \{s_t\}$.
3. Query expert to relabel all visited states: $\mathcal{D} \leftarrow \mathcal{D} \cup \{(s, \pi^*(s))\}_{s \in \mathcal{S}_{\text{visited}}}$.
4. Retrain $\pi_\theta$ on aggregated dataset.
- **Performance Bound**: Reduces error accumulation from $\mathcal{O}(T^2 \epsilon)$ to $\mathcal{O}(T \epsilon)$.
- **Bottleneck**: Requires continuous online access to an expert oracle.

---

## 2. Inverse Reinforcement Learning & GAIL (Section 11.2)

### 2.1 Maximum-Entropy IRL (Ziebart et al., 2008)
Resolves reward ambiguity by maximizing entropy over trajectory paths matching expert feature expectations:

$$p(\tau \mid r_\psi) = \frac{1}{Z(r_\psi)} \exp\left(\sum_t r_\psi(s_t, a_t)\right)$$

Soft policy formulation:
$$\pi(a \mid s) \propto \exp\left(Q^{\text{soft}}_{r_\psi}(s, a)\right)$$

Objective and gradient:
$$\max_\psi \; \mathcal{L}(\psi) = \sum_{\tau \in \mathcal{D}_{\text{expert}}} \left[\sum_t r_\psi(s_t, a_t)\right] - |\mathcal{D}_{\text{expert}}| \log Z(r_\psi)$$

$$\nabla_\psi \mathcal{L} = \mathbb{E}_{\tau \sim \text{expert}}\left[\sum_t \nabla_\psi r_\psi(s_t, a_t)\right] - \mathbb{E}_{\tau \sim p(\cdot \mid r_\psi)}\left[\sum_t \nabla_\psi r_\psi(s_t, a_t)\right]$$
- **Failure Mode**: Evaluating partition function $Z(r_\psi) = \int \exp(\sum_t r_\psi) d\tau$ in continuous or high-dimensional spaces requires an intractable inner RL loop.

### 2.2 GAIL (Generative Adversarial Imitation Learning)
Bypasses explicit reward estimation by treating occupancy measure matching as a minimax game:

$$\max_\phi \; \mathbb{E}_{(s, a) \sim \mathcal{D}_{\text{expert}}}\left[\log D_\phi(s, a)\right] + \mathbb{E}_{(s, a) \sim \pi_\theta}\left[\log (1 - D_\phi(s, a))\right]$$

Policy objective:
$$\min_\theta \; \mathbb{E}_{(s, a) \sim \pi_\theta}\left[\log (1 - D_\phi(s, a))\right] - \lambda \mathcal{H}(\pi_\theta)$$

Optimal discriminator ratio:
$$D^*_\phi(s, a) = \frac{p_{\text{expert}}(s, a)}{p_{\text{expert}}(s, a) + p_{\pi_\theta}(s, a)}$$

Implicit reward assigned to policy:
$$r^{\text{int}}(s, a) = -\log(1 - D_\phi(s, a)) = -\text{logsigmoid}(-z_\phi(s, a))$$

```python
# GAIL Core Alternating Update Loop
class GAIL:
    def update(self, n_policy_steps=5, n_disc_steps=1):
        # 1. Update Discriminator
        for _ in range(n_disc_steps):
            policy_s, policy_a = self.policy.sample_rollout()
            expert_logits = self.disc(self.expert_buffer.sample())
            policy_logits = self.disc(policy_s, policy_a)
            d_loss = (F.binary_cross_entropy_with_logits(expert_logits, torch.ones_like(expert_logits)) +
                      F.binary_cross_entropy_with_logits(policy_logits, torch.zeros_like(policy_logits)))
            self.disc_optim.zero_grad(); d_loss.backward(); self.disc_optim.step()

        # 2. Update Policy using surrogate reward
        for _ in range(n_policy_steps):
            s, a, s_next, _ = self.policy.rollout()
            with torch.no_grad():
                rewards = -F.logsigmoid(-self.disc(s, a))
            self.policy.ppo_update(s, a, rewards, s_next)
```

#### Comparison of Imitation Learning Paradigms:
| Dimension | Behavior Cloning (BC) | MaxEnt IRL | GAIL |
| :--- | :--- | :--- | :--- |
| **Compounding error / shift**| Suffers severely ($\mathcal{O}(T^2\epsilon)$)| Resolved ($\mathcal{O}(T\epsilon)$)| Resolved ($\mathcal{O}(T\epsilon)$)|
| **Requires environment model**| No | Yes (or soft-Q planner) | No (model-free rollouts) |
| **Explicit reward learned** | None | Yes (interpretable $r_\psi$) | No (implicit via discriminator) |
| **Computational cost** | Low (single supervised pass) | High (inner RL loop per outer step)| Medium (adversarial two-player game)|
| **High-dimensional scaling** | High | Poor | Moderate |

---

## 3. Meta-RL and Contextual Adaptation (Section 11.3)

### 3.1 Taxonomy of Meta-RL Adaptation
1. **Gradient-based (MAML)**: Finds an initialization $\theta^*$ from which single/few-step SGD adapts to new task distributions.
2. **Memory-based (RL² / SNAIL)**: Retains task transitions across episode boundaries within an RNN/LSTM hidden state $h_t$.
3. **Inference-based (PEARL)**: Estimates a latent task belief posterior $q_\phi(z \mid \tau)$ to condition an off-policy actor-critic.

### 3.2 MAML Mathematical Formulation
Inner-loop adaptation on task $T_i \sim p(T)$:
$$\theta_i' = \theta - \alpha \nabla_\theta \mathcal{L}_{T_i}(\theta)$$

Outer-loop meta-optimization across task distribution:
$$\min_{\theta} \; \mathbb{E}_{T_i \sim p(T)}\left[\mathcal{L}_{T_i}\left(\theta - \alpha \nabla_\theta \mathcal{L}_{T_i}(\theta)\right)\right]$$

Meta-gradient calculation (incorporating second-order Hessian):
$$\nabla_\theta \mathcal{L}_{T_i}(\theta_i') = \nabla_{\theta_i'} \mathcal{L}_{T_i}(\theta_i') \cdot \left(I - \alpha \nabla^2_\theta \mathcal{L}_{T_i}(\theta)\right)$$
- **FOMAML (First-Order MAML)**: Drops the Hessian term $\nabla^2_\theta \mathcal{L}_{T_i}(\theta)$ to avoid costly second derivatives.

### 3.3 PEARL Formulation
Latent task variable $z \sim q_\phi(z \mid \tau)$. Optimization objective with KL bottleneck:

$$\mathcal{L}_{\text{PEARL}} = -\mathbb{E}_{z \sim q_\phi}\left[\sum_t r(s_t, a_t, z)\right] + \beta \cdot D_{\text{KL}}\left(q_\phi(z \mid \tau) \,\|\, p(z)\right)$$

### 3.4 Algorithm Distillation (AD, Laskin et al., 2022)
Concatenates full multi-episode RL learning curves into Transformer context windows:
$$\text{Context} = [ (s_0, a_0, r_0, \ldots)_{\text{episode 1}}, (s_0, a_0, r_0, \ldots)_{\text{episode 2}}, \ldots, (s_0, a_0, r_0, \ldots)_{\text{episode N}} ]$$
- Trained entirely **offline** via autoregressive action prediction.
- Distills the RL algorithm itself directly into the causal attention activation weights.

| Dimension | RL² | Algorithm Distillation (AD) |
| :--- | :--- | :--- |
| **Architecture** | Recurrent neural net (LSTM/GRU) | Autoregressive causal Transformer |
| **Data source** | Online meta-training rollouts | Offline recorded learning histories |
| **Target learned** | Implicit task identification | The reinforcement learning algorithm itself |
| **Generalization** | Single task distribution | Cross-algorithm distillation (DQN, PPO, A2C) |

---

# Part III: Exploration, MARL, and Hierarchical RL (Chapter 14)

## 1. Intrinsic Motivation and Exploration (Section 12.1)

### 1.1 Sparse Reward Formulation
In hard-exploration environments (e.g., *Montezuma's Revenge*, *Pitfall*), uniform random exploration has probability $\approx 10^{-18}$ of reaching the first goal state. Intrinsic motivation augments external reward:

$$r^{\text{total}}_t = r^{\text{ext}}_t + \beta \cdot r^{\text{int}}_t$$

### 1.2 ICM (Intrinsic Curiosity Module, Pathak et al., 2017)
Decomposes exploration into feature learning, action prediction, and forward state prediction.
- **Inverse Model** $g_\phi$: Predicts action $a_t$ from $(\Phi(s_t), \Phi(s_{t+1}))$ to filter task-irrelevant ambient noise:
  $$\mathcal{L}_{\text{inv}}(\phi) = \text{CrossEntropy}(g_\phi(\Phi(s_t), \Phi(s_{t+1})), a_t)$$
- **Forward Model** $f_\psi$: Predicts next feature embedding from current state and action:
  $$\hat{\Phi}(s_{t+1}) = f_\psi(\Phi(s_t), a_t)$$
- **Intrinsic Reward & Overall Loss**:
  $$r^{\text{int}}_t = \frac{1}{2}\|\Phi(s_{t+1}) - \hat{\Phi}(s_{t+1})\|^2$$
  $$\mathcal{L} = \mathcal{L}_{\text{policy}}(\theta) + \lambda_{\text{inv}} \mathcal{L}_{\text{inv}}(\phi) + \lambda_{\text{fwd}} \mathcal{L}_{\text{fwd}}(\psi)$$
- **Failure Mode**: The **Noisy-TV problem**—uncontrollable stochastic transitions (e.g., white noise displays) produce perpetual high prediction error, trapping the agent.

### 1.3 RND (Random Network Distillation, Burda et al., 2018)
Uses two networks to evaluate novelty without learning an action model:
- Target network $\hat{f}(s)$: Randomly initialized, frozen parameters.
- Predictor network $f_\psi(s)$: Trained to match target outputs on visited states.

$$\mathcal{L}_{\text{RND}}(\psi) = \mathbb{E}_s\left[\|f_\psi(s) - \hat{f}(s)\|^2\right]$$
$$r^{\text{int}}(s) = \|f_\psi(s) - \hat{f}(s)\|^2$$
- **Noisy-TV Immunity**: Target function complexity is finite and deterministic; predictor error converges to zero even on stochastic patterns.

### 1.4 Multi-Timescale Novelty: NGU & Agent57
- **NGU (Never Give Up)**: Combines episodic novelty with life-long RND multiplicatively:
  $$r^{\text{int}}_t(s) = r^{\text{episodic}}_t(s) \cdot r^{\text{life-long}}_t(s)$$
  where episodic novelty uses a k-nearest neighbor distance kernel over stored episodic features:
  $$r^{\text{episodic}}_t = \frac{1}{\sqrt{k} + c \sum_{i=1}^{k} \frac{1}{\sqrt{d(s_t, s_i)}}}$$
- **Agent57**: Employs an adaptive multi-armed bandit meta-controller to dynamically select policy configurations $\{(\beta_i, \gamma_i, c_i)\}_{i=1}^{57}$ spanning pure exploitation to deep exploration, achieving super-human scores across all 57 Atari games.

---

## 2. Multi-Agent Reinforcement Learning (Section 12.2)

### 2.1 Nonstationarity & CTDE
In multi-agent systems, transition dynamics $P(s' \mid s, a_i, a_{-i})$ shift dynamically as co-agents update policies $\pi_{-i}$.
- **CTDE (Centralized Training with Decentralized Execution)**:
  - Decentralized actor: $\pi_i(a_i \mid o_i)$ executes purely on local observation $o_i$.
  - Centralized critic: $Q^{\text{tot}}(s, a_1, \ldots, a_n)$ or $V(s)$ utilizes global state and joint actions during training.

### 2.2 MADDPG (Multi-Agent DDPG, Lowe et al., 2017)
Continuous deterministic policy gradient for agent $i$:

$$\nabla_{\theta_i} J(\mu_{\theta_i}) = \mathbb{E}\left[\nabla_{\theta_i} \mu_{\theta_i}(o_i) \cdot \nabla_{a_i} Q_i(o_1, a_1, \ldots, o_n, a_n)\big|_{a_i = \mu_{\theta_i}(o_i)}\right]$$

Critic Bellman loss:
$$y_i = r_i + \gamma Q_i'(o'_1, a'_1, \ldots, o'_n, a'_n)\big|_{a'_j = \mu'_j(o'_j)}$$
$$\mathcal{L}(Q_i) = \mathbb{E}\left[(Q_i(o_1, a_1, \ldots, o_n, a_n) - y_i)^2\right]$$

### 2.3 MAPPO (Multi-Agent PPO, Yu et al., 2022)
Centralized value baseline $V_\phi(s)$ with clipped decentralized actor updates:

$$r_i^{(t)}(\theta_i) = \frac{\pi_{\theta_i}(a_i^{(t)} \mid o_i^{(t)})}{\pi_{\theta_{i,\text{old}}}(a_i^{(t)} \mid o_i^{(t)})}$$
$$\mathcal{L}_{\text{clip}}(\theta_i) = -\mathbb{E}\left[\min\left(r_i^{(t)}(\theta_i) A_i^{(t)}, \text{clip}(r_i^{(t)}(\theta_i), 1-\epsilon, 1+\epsilon) A_i^{(t)}\right)\right] - c_{\text{ent}} \mathcal{H}(\pi_{\theta_i})$$

```python
# MAPPO Centralized Critic and Clipped Policy Update
def mappo_update(actors, critic, buffer, n_agents, clip_eps=0.2):
    for epoch in range(E):
        for batch in buffer.iter():
            s, obs_list, a_list, old_logp_list, adv, ret = batch
            values = critic(s)
            for i in range(n_agents):
                new_logp = actors[i].log_prob(obs_list[i], a_list[i])
                ratio = (new_logp - old_logp_list[i]).exp()
                surr1 = ratio * adv[i]
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv[i]
                policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * actors[i].entropy().mean()
                actors[i].optimizer.step(policy_loss)
            value_loss = F.mse_loss(values, ret)
            critic.optimizer.step(value_loss)
```

### 2.4 Value Factorization: VDN & QMIX
- **VDN (Value-Decomposition Networks)**: Additive linear factorability:
  $$Q_{\text{tot}}(s, \mathbf{a}) = \sum_{i=1}^n Q_i(o_i, a_i)$$
- **QMIX**: Monotonicity constraint enabling decentralized argmax equivalence:
  $$\frac{\partial Q_{\text{tot}}}{\partial Q_i} \geq 0, \quad \forall i$$
  Implemented via hypernetworks generating non-negative mixing weights $W = |f_{\text{hyper}}(s)|$.

| Algorithm | Critic Input | Actor Input | On/Off-Policy | Core Target Domains |
| :--- | :--- | :--- | :--- | :--- |
| **IQL** | Local $o_i$ | Local $o_i$ | Off-policy | Decoupled baseline |
| **VDN / QMIX** | Global $s$ | Local $o_i$ | Off-policy | Cooperative discrete tasks |
| **MADDPG** | Joint $(o_1, a_1, \ldots, o_n, a_n)$| Local $o_i$ | Off-policy | Continuous mixed cooperative/competitive |
| **MAPPO** | Global $s$ | Local $o_i$ | On-policy | SMAC, Hanabi, Multi-Agent MuJoCo |

---

## 3. Hierarchical RL & World Models (Section 12.3)

### 3.1 Options Framework (Sutton, Precup, & Singh, 1999)
Defines temporally extended actions $\omega = (\mathcal{I}_\omega, \pi_\omega, \beta_\omega)$ within Semi-Markov Decision Processes (SMDP):
- Initiation set: $\mathcal{I}_\omega \subseteq \mathcal{S}$.
- Intra-option policy: $\pi_\omega(a \mid s)$.
- Termination probability: $\beta_\omega(s) \in [0, 1]$.

#### SMDP Bellman Equation:
$$Q^\mu(s, \omega) = \mathbb{E}\left[\sum_{t=0}^{T-1} \gamma^t r_t + \gamma^T \max_{\omega'} Q^\mu(s_T, \omega')\right]$$

### 3.2 FeUdal Networks (FuNs, Vezhnevets et al., 2017)
- **Manager**: Operates at temporal resolution $c$, outputting a latent directional goal vector $g_t \in \mathbb{R}^k$.
- **Worker**: Generates primitive actions conditioned on direction $g_t$.
- **Manager Cosine Direction Loss**:
  $$\mathcal{L}_M = -\langle g_t, \, \hat{z}_{t+c} - \hat{z}_t \rangle$$

### 3.3 HIRO (Data-Efficient HRL, Nachum et al., 2018)
- High-level policy outputs continuous displacement subgoal $g_t \in \mathbb{R}^d$ every $c$ steps.
- Low-level parameterized reward directly tracks target distance:
  $$r^l_t = -\|s_{t+1} - (s_t + g_t)\|$$
- **Goal Relabeling**: Re-indexes historical subgoals $g \to \tilde{g}$ to reflect what the updated low-level policy would have achieved, mitigating off-policy nonstationarity.

### 3.4 World Models & Genie Latent Actions
- **Genie (Bruce et al., 2024)**: Unsupervised world model trained on unlabeled video sequences.
- **Unsupervised Latent Action Discovery**:
  $$z_t = \text{LatentAction}(x_t, x_{t+1}), \quad x_{t+1} = \text{Decoder}(x_t, z_t)$$
  Allows policy learning entirely within neural generated environments without an explicit physics engine.

---

# Part IV: Synthesis & Failure Mode Taxonomy

```mermaid
graph TD
    subgraph Offline Distribution Shift
        A[Fixed Dataset D] --> B[Bellman Maximization max_a' Q]
        B --> C[OOD Actions Evaluated]
        C --> D[Extrapolation Error Amplified by 1/(1-gamma)]
        D --> E[Catastrophic Policy Overestimation]
    end

    subgraph Imitation Covariate Shift
        F[BC Supervised Loss on d_expert] --> G[Single Step Execution Error epsilon]
        G --> H[Enters Unvisited State s_ood]
        H --> I[Compounding Error O T^2 epsilon]
    end

    subgraph MARL Nonstationarity
        J[Independent Policy Updates] --> K[Environment Transitions P s'|s,a_i,a_-i Shift]
        K --> L[Stale Critic Targets & Coordination Collapse]
    end
```

### Critical Takeaways & Practical Recommendations
1. **Offline Value Regularization**: Always prefer IQL when stability on noisy/sparse continuous logs is paramount, as it completely circumvents OOD queries. CQL is preferred when sharp conservative bounds are strictly required.
2. **Offline Sequence Modeling**: Decision Transformer behaves as return-conditioned behavioral cloning; it cannot stitch suboptimal trajectory fragments. For trajectory stitching in offline datasets, use Diffuser or dynamic programming.
3. **Imitation Learning**: Use DAgger when interactive expert querying is viable ($\mathcal{O}(T\epsilon)$ error). If only demonstrations exist and sample efficiency is secondary, GAIL directly matches occupancy distributions.
4. **Exploration**: RND is substantially more robust than ICM in environments containing stochastic distractors (Noisy-TV). For deep exploration across varied domains, multi-timescale scheduling (NGU/Agent57) is necessary.
5. **Multi-Agent**: MAPPO is the most robust, hyperparameter-resilient baseline across cooperative discrete and continuous tasks, outperforming value factorization (QMIX) on modern benchmarks.

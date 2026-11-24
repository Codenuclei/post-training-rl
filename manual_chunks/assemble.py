chunks = [
    '/home/codenuclei/hands-on-modern-rl/manual_chunks/part1_classical_deep_rl.md',
    '/home/codenuclei/hands-on-modern-rl/manual_chunks/part2_offline_imitation_marl.md',
    '/home/codenuclei/hands-on-modern-rl/manual_chunks/part3_llm_alignment_reasoning.md',
    '/home/codenuclei/hands-on-modern-rl/manual_chunks/part4_agentic_multimodal_appendices.md',
]

header = """# Modern Reinforcement Learning: The Exhaustive Technical Compendium
## Comprehensive Multi-Agent Extraction Across All 187 Files (Chapters 1–32 & Appendices)
**Hands-On Modern Reinforcement Learning (WalkingLabs)**

---

## Master Architecture & Reading Map

```mermaid
flowchart TD
    subgraph Part1 ["Part I: Foundations & Deep RL (Ch 01-11)"]
        MDP["MDP 5-Tuple & Bellman"] --> DQN["Deep Q-Networks & Replay"]
        DQN --> PG["Policy Gradient Theorem & Baseline"]
        PG --> AC["Actor-Critic & Continuous Control"]
        AC --> PPO["PPO-Clip & GAE Advantage"]
    end

    subgraph Part2 ["Part II: Data Regimes & Complex Decision-Making (Ch 12-14)"]
        PPO --> Offline["Offline RL: CQL, IQL & Decision Diffusers"]
        Offline --> Imitation["Imitation Learning: BC, DAgger & GAIL"]
        Imitation --> MetaMARL["Exploration RND, MARL QMIX & Hierarchical"]
    end

    subgraph Part3 ["Part III: The LLM Post-Training & Reasoning Revolution (Ch 15-21)"]
        MetaMARL --> RLHF["Classical RLHF: Bradley-Terry & 4-Model PPO"]
        RLHF --> DPO["Direct Preference Optimization: DPO, KTO, SimPO"]
        DPO --> GRPO["Group Relative Policy Optimization: DeepSeek-R1 & veRL"]
        GRPO --> PRM["Process Supervision PRMs, MCTS & RLVR"]
    end

    subgraph Part4 ["Part IV: Frontier Multimodal, Physical & Industrial Scale (Ch 22-32 & Appendices)"]
        PRM --> Agents["Agentic SWE-Bench & Computer-Use"]
        Agents --> Multi["VLM Geometry EasyR1, Audio RVQ & VLA Robotics"]
        Multi --> Diffusion["Visual Diffusion MDPs & Self-Play Minimax"]
        Diffusion --> Infra["Industrial FSDP Memory Math & GPU Cluster Sizing"]
    end

    style Part1 fill:#eef2ff,stroke:#4f46e5
    style Part2 fill:#ede9fe,stroke:#6d28d9
    style Part3 fill:#fae8ff,stroke:#a21caf
    style Part4 fill:#dcfce7,stroke:#16a34a
```

---
"""

with open('/home/codenuclei/hands-on-modern-rl/MODERN_RL_MASTER_MANUAL.md', 'w') as out_f:
    out_f.write(header)
    for c in chunks:
        with open(c, 'r') as in_f:
            content = in_f.read()
            out_f.write('\n\n---\n\n' + content)

print("Master Compendium successfully assembled!")

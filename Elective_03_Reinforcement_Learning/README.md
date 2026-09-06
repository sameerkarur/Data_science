# Elective 03: Reinforcement Learning & Autonomous Decision Systems
**E&ICT Academy, IIT Kanpur — Specialization Syllabus & Engineering Guide**

---

## 📌 Domain Overview

Reinforcement Learning (RL) investigates how autonomous agents learn optimal behavioral policies through sequential trial-and-error interactions with dynamic environments. This elective covers fundamental Markov Decision Processes, Deep Q-Networks, Policy Gradients, Actor-Critic architectures, and modern Reinforcement Learning from Human Feedback (RLHF) used to align Large Language Models.

---

## 🧭 Specialization Architecture & Curriculum Roadmap

### 1. Foundations of Reinforcement Learning & MDPs
- **Formal MDP Formulation:** Five-tuple $(S, A, P, R, \gamma)$ defining states, actions, transition dynamics $P(s' | s, a)$, reward functions $R(s, a, s')$, and discount factors $\gamma \in [0, 1)$.
- **Value Functions & Bellman Equations:**
  $$V^\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t R_{t+1} \,\Big|\, S_0 = s \right]$$
  $$Q^\pi(s, a) = R(s, a) + \gamma \sum_{s' \in S} P(s' | s, a) V^\pi(s')$$
- **Dynamic Programming Solutions:** Policy Iteration (evaluation and improvement steps) and Value Iteration algorithms with convergence proofs.

### 2. Model-Free Value-Based Learning
- **Temporal-Difference (TD) Learning:** TD(0), SARSA (on-policy), and Q-Learning (off-policy) utilizing the TD-error $\delta_t = R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)$.
- **Deep Q-Networks (DQN):** Neural function approximation stabilized by:
  - *Experience Replay Buffer:* Breaking sequential temporal correlation and stabilizing sample distributions.
  - *Target Networks:* Mitigating moving-target optimization instability via Polyak parameter averaging.
  - *Dueling DQN & Double DQN (DDQN):* Separating state-value $V(s)$ and advantage $A(s, a)$, resolving overestimation bias.

### 3. Policy-Based & Actor-Critic Methods
- **Policy Gradient Theorem:** Direct gradient ascent on parameter vector $\theta$ optimizing expected trajectory rewards $J(\theta)$:
  $$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \nabla_\theta \log \pi_\theta(a_t | s_t) G_t \right]$$
- **Actor-Critic Frameworks (A2C / A3C):** Actor updates policy parameters $\pi_\theta(a|s)$ guided by Critic's value baseline $V_\phi(s)$ reducing variance.
- **Proximal Policy Optimization (PPO):** Clipped surrogate objective preventing destructively large policy updates:
  $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right]$$
  *(where $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$)*

### 4. RLHF & Preference Optimization in Modern LLMs
- **Reward Modeling:** Bradley-Terry preference model training reward predictor $r_\psi(x, y)$ on pairwise human comparisons:
  $$\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left(r_\psi(x, y_w) - r_\psi(x, y_l)\right) \right]$$
- **PPO Alignment with KL Penalty:** Constraining aligned policy $\pi_\theta$ within bounded KL-divergence from reference model $\pi_{\text{ref}}$:
  $$\max_\theta \mathbb{E}\left[ r_\psi(x, y) - \beta \, \mathbb{D}_{\text{KL}}\left(\pi_\theta(y|x) \,||\, \pi_{\text{ref}}(y|x)\right) \right]$$
- **Direct Preference Optimization (DPO):** Implicit mathematical re-parameterization eliminating the need for an explicit reward model or RL loop.

---

## 💻 Recommended Applied Projects & Research Benchmarks

1. **Autonomous Portfolio Trading Agent:** PPO-trained agent with transaction-cost modeling optimizing Sharpe ratio on historical equity order books.
2. **Robotic Manipulator Arm Control:** Soft Actor-Critic (SAC) continuous action space control in MuJoCo / Gymnasium simulations.
3. **LLM Alignment Pipeline:** DPO implementation aligning open-weights LLaMA-3 model on human preference datasets.

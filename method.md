# Method — A Study Guide for the Dynamic-Formation PPO with Teleop Project

> **Audience.** You. The presenter. This document is written so that you can
> walk into a room, give a 20-minute talk, and answer hard questions
> afterwards without needing to also read the code.
>
> **How to use it.** Read top-to-bottom once. Re-skim sections 5–8 the night
> before. Use the Q&A in section 12 as flashcards.

---

## Table of contents

1. [TL;DR](#1-tldr)
2. [Background concepts (beginner-friendly primer)](#2-background-concepts)
   - 2.1 [Reinforcement learning + MDPs](#21-reinforcement-learning--mdps)
   - 2.2 [Policy gradients (intuition + REINFORCE)](#22-policy-gradients)
   - 2.3 [PPO — the clipped surrogate objective](#23-ppo)
   - 2.4 [Generalised Advantage Estimation (GAE)](#24-gae)
   - 2.5 [Multi-agent RL — CTDE](#25-multi-agent-rl--ctde)
   - 2.6 [Graph Neural Networks](#26-graph-neural-networks)
   - 2.7 [Beta distribution as a bounded continuous policy](#27-beta-distribution-as-a-bounded-continuous-policy)
   - 2.8 [Hungarian algorithm](#28-hungarian-algorithm)
3. [The AFOR baseline](#3-the-afor-baseline)
4. [Our extension — problem statement](#4-our-extension--problem-statement)
5. [System architecture (file by file)](#5-system-architecture-file-by-file)
6. [The learning loop, step by step](#6-the-learning-loop-step-by-step)
7. [Reward design rationale](#7-reward-design-rationale)
8. [Design choices and tradeoffs](#8-design-choices-and-tradeoffs)
9. [Empirical results](#9-empirical-results)
10. [Known limitations](#10-known-limitations)
11. [Glossary](#11-glossary)
12. [Likely audience questions and model answers](#12-likely-audience-questions-and-model-answers)

---

## 1. TL;DR

We extended the **AFOR** paper (Coordinated Multi-Robot Navigation with
Formation Adaptation, [arXiv 2404.01618](https://arxiv.org/pdf/2404.01618))
so that a **single shared PPO policy** can drive a 4-robot cluster down a
hallway while **dynamically switching its target shape** based on how many
robots are currently under policy control:

- 4 active → **square**
- 3 active → **equilateral triangle**
- 2 active → **horizontal line**
- 1 active → **solo navigation**

A human can grab any robot at any moment with the keyboard (during the
demo) or a synthetic disturbance can grab it during training. The remaining
robots see the disturbance through a `teleop_mask` feature in their
observation and learn to **re-form** around the missing teammate, then
**re-absorb** it when it's released.

After 4648 PPO iterations on a single CPU, the policy reaches the goal
in **100% of 50 clean eval episodes** and **100% of 50 disturbed eval
episodes**, with formation error of only **1.6 cm** (clean) — far below
the agent radius of 8 cm.

The whole thing is built on a **graph neural network policy** (so that
the model is permutation-invariant across robots and naturally handles
varying neighbour counts) with a **Beta-distribution action head** (so
that the bounded velocity action space is sampled cleanly without
clipping artefacts) and **gradient masking** in the PPO loss (so that
teleop'd robots, whose actions are overridden by the human, do not
contribute spurious gradients).

---

## 2. Background concepts

If your audience is robotics-literate but RL-rusty, you may need to
spend the first 5 minutes on these. They are arranged in dependency
order: each subsection assumes the previous ones.

### 2.1 Reinforcement learning + MDPs

A **Markov Decision Process (MDP)** is the formal setting of RL. It's the tuple
`(S, A, P, R, γ)`:

| Symbol | Meaning | In this project |
|---|---|---|
| `S` | state space | positions, velocities, masks of 4 robots |
| `A` | action space | a 2D desired velocity per robot |
| `P(s' \| s, a)` | transition probability | the env's physics: kinematic integration with velocity + acceleration limits |
| `R(s, a)` | reward | sum of forward, formation, collision, stall, wall, goal terms |
| `γ` | discount factor | 0.995 — long-horizon, future rewards matter |

The **Markov property** says the next state depends only on the current
state and action — not on history. In our env, that's true if you put
velocity into the state (which we do).

A **policy** π(a | s) is a (possibly stochastic) mapping from states to
actions. The goal of RL is to find π* that maximises the expected
discounted return:

```
J(π) = E[ Σₜ γᵗ · R(sₜ, aₜ) ]    where actions come from π
```

### 2.2 Policy gradients

Two big families of RL algorithms:

- **Value-based** (Q-learning, DQN): learn the value of each (state, action) pair, act greedily. Works well for discrete actions.
- **Policy-gradient**: parametrise the policy directly (πθ), update θ by gradient ascent on J. Works well for continuous actions — which is our case.

The **policy gradient theorem** says:

```
∇θ J(πθ) = E[ ∇θ log πθ(a|s) · Aπ(s, a) ]
```

i.e. push up the log-probability of actions whose **advantage** Aπ(s,a)
is positive (better than average), push down those whose advantage is
negative. The advantage is just "this action's return minus what we'd
have expected on average from this state".

The simplest implementation is **REINFORCE**: roll out an episode, sum
the discounted rewards, multiply by `∇log π`. Simple but very high
variance. Modern policy-gradient methods (TRPO, PPO) build on this with
two key additions: **a baseline (value function) to reduce variance**,
and **a constraint to prevent the policy from updating too far in one
step**.

### 2.3 PPO

**Proximal Policy Optimization** ([Schulman et al. 2017](https://arxiv.org/abs/1707.06347))
is the workhorse of modern continuous-control RL. The core idea: do
**multiple gradient steps per batch** for sample efficiency, but cap
how far the new policy can drift from the old one to prevent
catastrophic updates.

The **clipped surrogate objective** is:

```
L^{CLIP}(θ) = E[ min( rₜ(θ) · Â,  clip(rₜ(θ), 1-ε, 1+ε) · Â ) ]

where rₜ(θ) = πθ(aₜ|sₜ) / πθ_old(aₜ|sₜ)
```

`rₜ(θ)` is the **probability ratio** — how much more likely the new
policy makes the action than the old policy did. If `rₜ` drifts outside
`[1-ε, 1+ε]` (we use ε = 0.2), the gradient is clipped so further
updates in that direction don't help. This bounds the per-update KL
divergence between the new and old policies.

The full PPO loss adds a **value-function loss** (so the critic learns
to predict the return) and an **entropy bonus** (so the policy doesn't
collapse to deterministic too quickly):

```
L^{total} = L^{CLIP} − c_v · L^{value} + c_h · H[πθ]
```

In our trainer (`code/train_hallway.py:298-301`):

```python
loss = pg_loss − config["entropy_coeff"] · ent_loss
        + v_loss · config["vf_loss_coeff"]
```

Coefficients: `entropy_coeff = 0.001`, `vf_loss_coeff = 1.0`.

### 2.4 GAE

Computing a clean advantage `Â = Q(s,a) − V(s)` requires knowing the
true Q value, which we don't. **Generalised Advantage Estimation**
([Schulman et al. 2015](https://arxiv.org/abs/1506.02438)) interpolates
between two extremes:

- **One-step TD** (low variance, high bias): `δₜ = rₜ + γ V(sₜ₊₁) − V(sₜ)`
- **Monte Carlo return** (high variance, no bias): `Σ γᵏ rₜ₊ₖ − V(sₜ)`

GAE blends them with a parameter λ ∈ [0, 1]:

```
Âₜ = δₜ + (γλ) · δₜ₊₁ + (γλ)² · δₜ₊₂ + …
```

We use λ = 0.95 — closer to Monte Carlo, but with enough TD smoothing
to keep variance manageable. In our trainer, GAE runs **backward
through the rollout** (`code/train_hallway.py:248-260`):

```python
for t in reversed(range(T)):
    nextnonterminal = 1.0 - dones_buf[t + 1]   # 0 if episode ended at t+1
    nextvalues = values_buf[t + 1]
    delta = rewards_buf[t] + γ · nextvalues · nextnonterminal − values_buf[t]
    advantages[t] = lastgaelam = delta + γ · λ · nextnonterminal · lastgaelam
returns_buf = advantages + values_buf
```

Going backwards lets us compute each step's advantage in O(1) using
the next step's already-computed `lastgaelam`.

### 2.5 Multi-agent RL — CTDE

Two extreme architectures for multi-agent learning:

- **Independent learners** — each robot has its own policy, treats
  others as part of the environment. Brittle: as the other robots
  learn, the env distribution shifts.
- **Joint policy** — one policy that outputs all robots' actions
  jointly. Action space grows exponentially with N.

The middle ground used here is **CTDE — Centralised Training,
Decentralised Execution**. We have one *shared* policy π that runs on
each robot independently at inference, but during training we
backpropagate through the joint experience of all robots in all
parallel envs. Parameter sharing is what makes the GNN policy below
practical: it's the *same network* every robot uses.

### 2.6 Graph Neural Networks

A GNN treats the robots as nodes in a graph, with edges between robots
that are within communication range (we use `comm_range = 2.0 m`). At
each layer, a node updates its features by aggregating messages from
neighbours.

The simplest **message-passing layer** is:

```
hᵢ' = AGG{ message(hᵢ, hⱼ) : j ∈ neighbours(i) }
```

Our `ModGNNConv` (`code/model.py:11-30`) uses `aggr="add"` (sum
neighbour messages) and a learnable message function:

```python
def message(self, x_i, x_j):
    return self.nn(x_j - x_i)   # message is a function of the relative feature
```

Three properties make this perfect for multi-robot formation:

1. **Permutation invariant** — relabelling robots doesn't change the
   policy output. Crucial because robots have no canonical identity.
2. **Variable neighbour count** — each robot's update works with any
   number of neighbours.
3. **Local computation** — robot `i`'s update only depends on robots
   within `comm_range`, mimicking real radio constraints.

The full network in `model.py` is:

```
   per-robot raw features (8-dim, see §5)
                ↓
        Encoder MLP (linear → ReLU → … → linear)
                ↓
        ModGNNConv (one layer of message passing within comm_range)
                ↓
        Post MLP (linear → ReLU → … → linear)
                ↓
   per-robot output (4-dim for actor: alpha+beta of Beta dist on 2D action,
                     1-dim for critic)
```

There are **two separate** GNN branches with the same architecture but
different parameters: one for the actor, one for the critic
(`code/model.py:111-112`).

### 2.7 Beta distribution as a bounded continuous policy

For continuous actions in `[low, high]`, the obvious choice is a
**Gaussian** — but the Gaussian's support is `(−∞, +∞)`. You then have
to clip the sampled action to the action bounds, which:

- biases the gradient (the clipped action has a different log-prob than what was sampled),
- creates dead-zones at the boundaries (lots of probability mass piles up at the edges),
- doesn't compose nicely with the Beta-style "encourage extremes vs middle" of robotics tasks.

The **Beta(α, β)** distribution has support `[0, 1]` natively. We take
the network's output `(α, β)` per action dimension, sample `u ~ Beta(α, β)`,
and squash to `[low, high]` by `a = u · (high − low) + low`.

In `code/model.py:185-196`:

```python
agent_probs = torch.distributions.Beta(concentration1=alpha, concentration0=beta)
agent_action_normalised = agent_probs.rsample()    # in [0, 1]
agent_logp = agent_probs.log_prob(agent_action_normalised).sum(-1)
agent_action = agent_action_normalised · (high − low) + low
```

We constrain `α, β > 1` (via a softplus + 1 transform on
`code/model.py:173-174`) so the distribution stays unimodal. This is
[Chou et al. 2017](https://arxiv.org/abs/1702.05033)'s recipe and works
notably better than Gaussian + tanh for bounded action spaces.

### 2.8 Hungarian algorithm

Given a cost matrix `C ∈ ℝⁿˣⁿ` where `C[i, j]` is the cost of assigning
worker `i` to slot `j`, the **Hungarian algorithm** finds the
permutation `σ` that minimises `Σᵢ C[i, σ(i)]`. Runtime: O(n³).

We use it in `env_hallway._formation_reward` to assign the `k` active
robots to the `k` slots of the target formation. Without optimal
assignment, fixed-index slot assignment would punish a configuration
like "robots 1 and 3 swapped" with huge formation error even though
the cluster *visually* looks correct. With Hungarian assignment, the
reward is permutation-invariant — exactly the right inductive bias to
match the GNN policy.

`scipy.optimize.linear_sum_assignment` is the standard
implementation; for `n = 4` it's effectively free.

---

## 3. The AFOR baseline

The original AFOR paper trains a PPO policy that drives **5 robots in
a fixed formation** (line, pentagon, or wedge — pick one at training
time) through an obstacle gauntlet. The repo ships three near-identical
env files, one per formation:

- `code/env_line.py` — robots in a horizontal line
- `code/env_pentagon.py` — robots on a pentagon
- `code/env_wedge.py` — wedge/V-shape

Each env file:

- Has `n_agents = 5` baked in.
- Encodes the target formation as hardcoded relative positions in
  config (`agent_formation`).
- Computes formation reward by **fixed-index distance penalties**:
  e.g. `rewards[:, 4] -= 2 · |y_4 - y_2|` to keep robot 4 aligned with
  robot 2 (`code/env_line.py:316`).
- Hardcodes the obstacle layout — staggered walls forming a passage.

The trainer (`code/train.py`) imports one of these envs and runs PPO
with `n_agents = 5` baked into the buffer shapes and the model
(`code/model.py:99` originally read `self.n_agents = 5`).

**Capabilities of the baseline:**
- ✓ Multi-robot formation control with PPO.
- ✓ Obstacle avoidance via the wall-collision penalty.
- ✓ Permutation-invariant GNN policy.
- ✓ Beta-distribution action head.

**What it cannot do:**
- ✗ Vary the active cluster size — `n_agents` is fixed at 5.
- ✗ Switch target formations at runtime.
- ✗ Handle a robot being pulled out of formation by external control.
- ✗ Re-form around a missing or returning teammate.

These four things are exactly what we add.

---

## 4. Our extension — problem statement

**Setting.** A long, narrow, obstacle-free hallway (2 m × 12 m). Four
robots spawn near the south end (`y = -5`), goal line at `y = +5`. The
cluster must traverse the hallway while staying in formation.

**The twist.** At any moment, any robot can be **teleop'd** — its action
is overridden by a human (or by a synthetic disturbance during training)
and the remaining robots must adapt their target shape:

- 4 active → square (the default at episode start)
- 3 active → equilateral triangle
- 2 active → horizontal line
- 1 active → solo (no formation, just navigate)

When the teleop'd robot is released, it rejoins the cluster and the
formation reverts.

**Why this is non-trivial.**

1. The target shape changes mid-episode. The policy needs to *condition*
   on the current cluster size, which it can't see directly — it has to
   infer it from the `teleop_mask` feature and the positions of the
   other robots.
2. The policy must avoid colliding with the teleop'd robot even though
   that robot's behaviour is unpredictable (it's a human / random
   sinusoid).
3. Reward credit for the teleop'd robot is meaningless (a human is
   driving it, not the policy). Naive PPO would attribute that robot's
   reward to the policy gradient and corrupt training. We solve this
   with **gradient masking** in the PPO loss (see §6).

**Approach overview.**

| Component | What it does |
|---|---|
| `FormationHallwayEnv` | env with up to 4 robots, dynamic target shape, teleop interface |
| `RandomTeleop` | synthetic disturbance for training: occasionally grabs a robot, drives it sideways, releases it |
| `KeyboardTeleop` | for the demo: 1-4 toggle, WASD drive |
| Patched `model.py` | parametric `n_agents`, optional 8-dim per-robot input including the masks |
| `train_hallway.py` | PPO trainer with **per-robot loss masking** by `(1 - teleop_mask)` |
| `metrics.py` + `runs/<ts>/` | persistent CSV/JSONL/JSON for cross-run comparison |

---

## 5. System architecture (file by file)

The codebase has two layers: **baseline** (untouched: `env_line.py`,
`env_pentagon.py`, `env_wedge.py`, `train.py`, `eval.py`) and **new**
(everything below). One file (`model.py`) was patched in place to be
parametric — the baseline still uses it without changes.

```
afor/
├── plan.md
├── README.md                     # operational manual
├── method.md                     # this file
├── code/
│   ├── contract.py               # NEW · constants + interface contract
│   ├── env_hallway.py            # NEW · the env
│   ├── teleop.py                 # NEW · RandomTeleop + KeyboardTeleop
│   ├── metrics.py                # NEW · RunLogger + EpisodeAccumulator
│   ├── checkpoint.py             # NEW · save/load helpers (resume support)
│   ├── train_hallway.py          # NEW · PPO trainer with masked loss
│   ├── eval_hallway.py           # NEW · headless / rendered eval
│   ├── render_hallway.py         # NEW · pygame renderer
│   ├── run_demo.py               # NEW · interactive demo binary
│   ├── scripts/
│   │   └── compare_runs.py       # NEW · cross-run comparison
│   ├── model.py                  # PATCHED · parametric n_agents + use_masks
│   ├── env_line.py               # baseline (unchanged)
│   ├── env_pentagon.py           # baseline (unchanged)
│   ├── env_wedge.py              # baseline (unchanged)
│   ├── train.py / eval.py        # baseline (unchanged)
│   └── README.md                 # baseline-specific instructions
├── tests/
│   ├── conftest.py
│   ├── fake_env.py               # contract-shaped stub
│   └── test_formation.py         # 9 unit tests
└── runs/                         # gitignored
    └── 20260503_195621_hallway-v1/
        ├── config.json
        ├── iterations.csv
        ├── episodes.jsonl
        ├── eval_clean.json       # added by you after evaluation
        ├── eval_teleop.json
        └── weights/
            ├── weights_epoch200.pt … weights_epoch4600.pt
            └── latest.pt → weights_epoch4600.pt
```

### 5.1 `code/contract.py` — the shared interface

A single file every other module imports from:

```python
MAX_AGENTS = 4
DT = 0.05
WORLD_W = 2.0
WORLD_H = 12.0
SPAWN_Y = -5.0
GOAL_Y = +5.0
MAX_V = 1.0
MAX_A = 2.0
AGENT_RADIUS = 0.08
FORMATION_SCALE = 0.35

REWARD_COEFFS = {
    "k_fwd": 5.0,    "k_stall": 0.5,    "k_form": 2.0,
    "k_coll": 5.0,   "k_wall": 1.0,     "k_goal": 20.0,
    "stall_window": 20,                  "stall_eps": 0.02,
}
```

Why a separate file: changing any of these constants (e.g. shrinking
`FORMATION_SCALE`) ripples into the env, the renderer, and the
analyses. One source of truth = no drift.

### 5.2 `code/env_hallway.py` — the env

Subclasses `gym.Env`. Mirrors the existing `PassageEnv` interface
(vector_reset / vector_step / reset_at) so the existing PPO trainer
scaffolding plugs in directly.

**State** (per env, all torch tensors of shape (num_envs, MAX_AGENTS, ...)):

| Attribute | Shape | Meaning |
|---|---|---|
| `ps` | `(nE, 4, 2)` | positions in metres |
| `measured_vs` | `(nE, 4, 2)` | velocities |
| `goal_ps` | `(nE, 4, 2)` | broadcast goal `(0, GOAL_Y)` |
| `teleop_mask` | `(nE, 4)` | 1.0 if under teleop, else 0.0 |
| `present_mask` | `(nE, 4)` | always 1.0 in v1, reserved |
| `teleop_vels` | `(nE, 4, 2)` | override velocities for teleop'd robots |
| `timesteps` | `(nE,)` | step count per env |
| `goal_reached` | `(nE,)` | bool, one-shot |

**Action** (input to `vector_step`): numpy array `(nE, 4, 2)` — desired
2D velocity per robot. The env then **overrides slots where
`teleop_mask == 1`** with the stored teleop velocity:

```python
# code/env_hallway.py:241
mask3 = self.teleop_mask.unsqueeze(-1)
actions_t = actions_t * (1.0 - mask3) + self.teleop_vels * mask3
```

**Dynamics.** Standard kinematic integration with velocity and
acceleration limits (lifted from `env_line.py`):

```python
# code/env_hallway.py:243-247
desired_vs = clip(actions_t, -MAX_V, MAX_V)
desired_as = (desired_vs - self.measured_vs) / dt
possible_as = clip(desired_as, MIN_A, MAX_A)
possible_vs = self.measured_vs + possible_as * dt
```

Then per-robot collision check: trial-step each robot in turn; if it
would collide with another robot, don't apply that step and dock it
the collision penalty (`code/env_hallway.py:255-263`). Then wall
clamping in x.

**Observation** (`get_obs`, returned as a list of dicts, one per env):

```python
{
  "pos":          [[x,y]] * 4,
  "vel":          [[vx,vy]] * 4,
  "goal":         [[0,5]] * 4,
  "teleop_mask":  [0,0,1,0],     # one-hot of currently-teleop'd
  "present_mask": [1,1,1,1],
  "time":         [[t]] * 4,
}
```

**Target formation.** Pure function `target_formation_positions(n)`
returns `(n, 2)` torch tensor of slot positions, centred at origin:

- 4: vertices of a square `(±s/2, ±s/2)` with side `s = FORMATION_SCALE`
- 3: equilateral triangle, one vertex pointing +y
- 2: `[(-s/2, 0), (s/2, 0)]` — horizontal line
- 1: `[(0, 0)]`

These are unit-tested in `tests/test_formation.py`.

**Step infos.** In addition to the per-agent reward, each step's info
dict includes diagnostic fields the trainer plumbs into the metrics
logger: `active_count`, `goal_reached`, `formation_error`,
`fwd_velocity`, `stalled`, `wall_hit`, `collided`. See §7 for what
each reward term computes.

### 5.3 `code/model.py` — encoder → GNN → policy + value heads

The model was patched to be parametric. Two knobs from cfg now:

- `obs_space["pos"].shape[0]` → `n_agents` (was hardcoded 5)
- `cfg["use_masks"]: bool` → if True, per-robot input grows from 6 to 8 dims

The per-robot input vector is built in `Model.forward`
(`code/model.py:115-128`):

```python
feats = [goal - pos, pos, pos + vel]    # 6 dims
if self.use_masks:
    feats += [teleop_mask.unsqueeze(-1), present_mask.unsqueeze(-1)]   # +2
x = torch.cat(feats, dim=-1)            # (bs, n_agents, 6 or 8)
```

Then through the encoder MLP (input → 16 → 32 → 32 → `msg_features=32`),
the `ModGNNConv` layer, and the post MLP (64 → 64 → 64 → 4) which
produces `(α₁, β₁, α₂, β₂)` per robot for the 2D Beta action head.

The value branch is identical except the post MLP outputs a single
scalar per robot.

Backward-compatibility note: the existing `env_line.py` baseline still
loads this model with `use_masks=False` and the original 6-dim input,
verified by the smoke check in §9 of plan.md.

### 5.4 `code/teleop.py` — RandomTeleop + KeyboardTeleop

`RandomTeleop` runs **per env** during training. Internal state is one
`_Grab` object per env (or None). Each call to `step()`:

1. If no grab is active: with prob `p_grab = 0.005`, pick a random
   robot, sample a duration in `[40, 160]` steps, choose a left/right
   drift direction. Set `env.set_teleop(e, robot, True)`.
2. If a grab is active: drive the robot with a sinusoidal lateral
   velocity `vx = drift_speed · drift_dir · cos(0.15 · age)` plus a
   small `base_vy`. With prob `p_release = 0.01` per step or once the
   duration elapses, call `env.set_teleop(e, robot, False)`.

Critical limitation: **only ONE robot per env can be grabbed at a
time**. This shapes what the policy sees during training — see §10.

`KeyboardTeleop` is the inference-time driver:

- `1` / `2` / `3` / `4` toggle teleop on the corresponding robot.
- `WASD` set the velocity of the *most recently selected* teleop
  robot. Other teleop'd robots are held in place (zero velocity).
- `0` releases all.

Both classes call exactly `env.set_teleop(e, r, on/off)` and
`env.set_teleop_action(e, r, vel)` — that's the whole interface.

### 5.5 `code/metrics.py` — RunLogger + EpisodeAccumulator

`RunLogger` writes three append-only files per run:

- `config.json` — one-shot snapshot at the start of training: PPO
  hyperparameters, reward coefficients, env constants, teleop params,
  the full argparse `args`, and a `meta` block with timestamp,
  hostname, git SHA, torch version. This is the "what was this run?"
  file.
- `iterations.csv` — one row per PPO iteration with columns
  `iter, wall_time_s, env_steps, policy_loss, value_loss, entropy,
  total_loss, approx_kl, clip_frac, grad_norm, mean_reward,
  mean_episode_length, lr`. This is the *primary* learning-curve
  artefact — opens in pandas, Excel, anything.
- `episodes.jsonl` — one JSON object per finished episode, with
  per-cluster-size formation error, collisions, teleop grabs, etc.

`EpisodeAccumulator` is the per-env counter. The trainer ticks it
each step and emits to `RunLogger` whenever `done` fires.

**Why CSV/JSONL not tensorboard:** plain text diffs cleanly across
runs, doesn't require a viewer process, and the JSONL records can
grow new fields without breaking historical files.

### 5.6 `code/checkpoint.py` — save/load helpers

The new checkpoint format is `{"agent": state_dict, "optimizer": state_dict, "iteration": int}`
so a run can resume with optimizer state intact. `load_checkpoint`
also accepts the legacy bare-state_dict format, so older AFOR
checkpoints still load (without optimizer state). This format is
shared by `eval_hallway.py`, `run_demo.py`, and `train_hallway.py`'s
`--resume` flag.

### 5.7 `code/train_hallway.py` — the trainer

The biggest file. Walked through step-by-step in §6.

### 5.8 `code/render_hallway.py` — pygame renderer

Top-down view of the hallway. World y is "forward" so screen y is
inverted. Draws:

- hallway walls and goal line
- 4 colored circles (one per robot)
- a red "TELEOP" ring around teleop'd robots
- faded outline of the **target formation slots** at the active-cluster
  centroid, Hungarian-assigned to the actual robot positions
- a HUD line with active count, episode step, and accumulated reward

`--self-test` cycles through `n_active ∈ {4, 3, 2, 1}` with hardcoded
poses and writes a 4-panel PNG so the renderer can be sanity-checked
headless.

### 5.9 `code/eval_hallway.py` and `code/run_demo.py`

Both load a checkpoint via `load_checkpoint` and roll the policy
against the env. `eval_hallway.py` is headless by default and writes
a top-level summary plus per-episode records to `eval.json`.
`run_demo.py` opens a pygame window, attaches `KeyboardTeleop`, and
steps at real-time speed (`clock.tick(1/DT) = 20 Hz`).

---

## 6. The learning loop, step by step

This is the heart of the system. One **PPO iteration** consists of a
**rollout phase** (collect experience) followed by an **update phase**
(do gradient steps on that experience). After 4648 such iterations,
we had a converged policy.

The trainer's main loop is `code/train_hallway.py:189`. Let's
annotate it.

### 6.1 Setup (once, before the loop)

```python
env = FormationHallwayEnv(config["env_config"])     # 16 parallel envs
agent = Agent(env, config).to(device)
optimizer = optim.Adam(agent.parameters(), lr=5e-5, eps=1e-5)
teleop = RandomTeleop(env, ...)                     # one shared instance
logger = RunLogger("runs/", tag="hallway-v1")

# pre-allocated buffers, sized for one rollout's worth of data
# T = max_steps = 600, nE = num_envs = 16, nA = MAX_AGENTS = 4
actions_buf  = zeros(T, nE, nA, 2)
logprobs_buf = zeros(T, nE, nA)
rewards_buf  = zeros(T, nE, nA)
dones_buf    = zeros(T, nE)
values_buf   = zeros(T, nE, nA)
teleop_buf   = zeros(T, nE, nA)   # so we can mask the loss later
```

### 6.2 Rollout phase (the inner loop, T = 600 steps)

For each of the 600 timesteps:

1. **Observe.** Call `agent.format_input(next_obs, device)` to convert
   the list-of-dicts (one per env) into a batched tensor dict.
2. **Sample action.** With `torch.no_grad()` (we're collecting, not
   training), call `agent.get_action_and_value(x)`. This:
   - Runs `Model.forward(x)` to get `(α, β)` per robot per env.
   - Builds a Beta distribution and samples an action `u` in `[0, 1]`.
   - Squashes to `[-MAX_V, MAX_V]`.
   - Returns `(action, logprob, entropy, value)`.
3. **Apply teleop.** Call `teleop.step()`. This may set
   `env.teleop_mask[e, r] = 1` for some robot in some env, and write
   a sinusoidal `teleop_vel` for it.
4. **Step the env.** `env.vector_step(action.cpu().numpy())`. The env
   overrides teleop'd slots with the stored teleop_vel, runs the
   kinematic update, computes per-robot rewards.
5. **Store everything.** Append to `actions_buf, logprobs_buf,
   values_buf, rewards_buf, dones_buf, teleop_buf` at index `step`.
6. **Update accumulators.** For each env, tick the
   `EpisodeAccumulator` with this step's per-agent rewards, formation
   error, fwd velocity, etc.
7. **Handle done.** If `done[e]`, emit the accumulator's payload to
   `episodes.jsonl`, reset the env at `e`, reset the teleop state for
   that env, and reset the accumulator.

After 600 steps × 16 envs = **9600 env-steps of experience** are
sitting in the buffers, with episode boundaries marked in `dones_buf`.

### 6.3 GAE — going backwards in time

Before running gradient updates we need to compute the **advantage**
of every action that was taken — a measure of "how much better than
average was this?". GAE does that by walking backwards through the
buffer:

```python
# code/train_hallway.py:248-260
with torch.no_grad():
    next_value = agent.get_value(...)            # bootstrap from final state
    advantages = zeros_like(rewards_buf)
    lastgaelam = 0.0
    for t in reversed(range(T)):
        nextnonterminal = 1.0 - dones_buf[t + 1]   # 0 if episode ended
        nextvalues = values_buf[t + 1]
        delta = rewards_buf[t] + γ · nextvalues · nextnonterminal − values_buf[t]
        advantages[t] = lastgaelam = delta + γ · λ · nextnonterminal · lastgaelam
    returns_buf = advantages + values_buf
```

Two subtle things here. First, `nextnonterminal` zeros out the
bootstrap whenever the next step starts a new episode — otherwise
we'd leak future-episode value estimates back into the current
episode. Second, GAE *aggregates per-robot per-env*, so the advantage
tensor is shape `(T, nE, nA)`.

### 6.4 PPO update with the gradient mask — the key ML innovation here

Now we do `num_sgd_iter = 8` epochs of gradient descent over the
collected data:

```python
# code/train_hallway.py:266-303
for epoch in range(8):
    shuffle b_inds       # b_inds is a permutation of [0, T)
    for mb_t in b_inds:  # one timestep per minibatch
        _, newlogprob, entropy, newvalue = agent.get_action_and_value(
            obs_per_step[mb_t], actions_buf[mb_t]
        )
        ratio = (newlogprob - logprobs_buf[mb_t]).exp()    # (nE, nA)

        mb_adv = advantages[mb_t]
        if config["norm_adv"]:
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

        # ⭐ The masking step — gradients flow only through policy-controlled robots
        active = 1.0 - teleop_buf[mb_t]                    # (nE, nA), 0 for teleop'd
        norm = active.sum().clamp(min=1.0)

        pg_loss1 = -mb_adv * ratio
        pg_loss2 = -mb_adv * clamp(ratio, 1 - ε, 1 + ε)
        pg_loss = (max(pg_loss1, pg_loss2) * active).sum() / norm   # masked + normalised

        v_unclipped = (newvalue - returns_buf[mb_t]) ** 2
        v_clipped = values_buf[mb_t] + clamp(newvalue - values_buf[mb_t], -1.0, 1.0)
        v_clipped_loss = (v_clipped - returns_buf[mb_t]) ** 2
        v_loss = 0.5 * (max(v_unclipped, v_clipped_loss) * active).sum() / norm

        ent_loss = (entropy * active).sum() / norm
        loss = pg_loss - 0.001 * ent_loss + 1.0 * v_loss

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(agent.parameters(), 0.5)
        optimizer.step()
```

The `active` mask is the new-versus-baseline ML trick. Without it,
teleop'd robots would receive bogus advantage signals (they didn't
take the policy's action, they took the human's), and the value head
would learn to predict the human's reward as if it were the policy's
fault. With the mask, those slots are zeroed before the loss reduces,
and the normalisation by `active.sum()` keeps the loss scale stable
regardless of how many robots are teleop'd.

### 6.5 Iteration end: log + checkpoint

After the 8 SGD epochs, the trainer:

- Computes `mean_reward_iter = total_reward / global_step` (a running
  average — note this is a cumulative ratio, not a per-iteration mean,
  which is a known bug in the headline number; the per-iteration
  truth is the `episodes.jsonl` analysis we ran in §9).
- Calls `logger.log_iter(...)` to append a row to `iterations.csv`.
- Prints a one-line summary.
- Every `checkpoint_every = 200` iterations (and on the last
  iteration), saves `{agent, optimizer, iteration}` and updates the
  `weights/latest.pt` symlink.

Then we go back to step 6.2 with the next `iteration`. The env
**does not reset** between iterations — each rollout picks up from
wherever the env was at the end of the previous one. This is
important for sample efficiency: episodes can span iteration
boundaries.

---

## 7. Reward design rationale

Every reward term lives in `code/contract.py::REWARD_COEFFS` and is
applied in `code/env_hallway.py::vector_step`. Each is justified
below: what it incentivises, what would go wrong without it, and how
the coefficient was chosen.

| Term | Coeff | Formula | Rationale |
|---|---|---|---|
| **Forward progress** | `k_fwd = 5.0` | `+ k_fwd · dy` per step | The dominant per-step shaping signal. Without it, the policy has no early gradient direction; reaching the goal is too rare to discover from scratch. Coeff calibrated so 1 sec of full-speed forward motion ≈ +0.25 reward, accumulating to +50 over a successful episode. |
| **Stall penalty** | `k_stall = 0.5` | `- k_stall` if cluster centroid moved < ε over last K steps | Without this, a policy can locally maximise reward by hovering near a high-reward area — e.g. holding formation perfectly while not moving. Penalty triggers if the centroid moved less than 2 cm over 20 steps (=1 second at 20 Hz). |
| **Formation error** | `k_form = 2.0` | `- k_form · dist_to_assigned_slot` (Hungarian) | Pulls each robot toward its target slot. Hungarian assignment makes it permutation-invariant. Coeff calibrated so a 10 cm formation error costs 0.2/step ≈ 40 over an episode — comparable to the forward-progress reward, so neither dominates. |
| **Inter-robot collision** | `k_coll = 5.0` | `- k_coll` if any robot pair within `2 · AGENT_RADIUS` | Hard constraint expressed as a sharp penalty. Large enough that one collision wipes out several seconds of forward progress. Without it the formation can collapse onto a single point. |
| **Wall overshoot** | `k_wall = 1.0` | `- k_wall · |overshoot_x|` | Soft barrier on the hallway walls. Linear in overshoot, so the policy can briefly graze a wall but pays for it. Combined with the env hard-clamping x to the wall, this creates a smooth gradient pulling robots toward the centre line. |
| **Goal bonus** | `k_goal = 20.0` | `+ k_goal` once cluster centroid passes `GOAL_Y` (one-shot, episode terminates) | Sharp positive signal for the macro objective. One-shot to prevent "dancing on the goal line" exploits. Coeff is somewhat conservative — see §10, where bumping to 50 may help early training. |

Two rewards apply only to **active** robots (`teleop_mask[i] == 0`):
forward progress and goal bonus are zeroed for teleop'd robots inside
the env (`code/env_hallway.py:289`), and *all* terms get masked again
in the trainer's loss for double safety.

---

## 8. Design choices and tradeoffs

For each non-obvious decision, here is what we did, what we
considered, and why we picked the way we did.

### 8.1 Single shared policy vs three policies (one per cluster size)

**Chose:** one shared policy, conditioned on `teleop_mask`.

**Alternative:** three completely separate networks, one per active
cluster size.

**Why ours wins:** parameter sharing across configurations is a
strong inductive bias — the policy learns a *general* "stay close to
your assigned slot" behaviour, then specialises via the mask
features. Separate networks would each see less data and have to
re-learn the basics. Plus the GNN is already permutation-invariant,
so the same architecture handles all sizes naturally.

**Tradeoff:** the single policy must learn to interpret the mask.
We pay 2 input dims for that. Cheap.

### 8.2 GNN vs MLP vs Transformer

**Chose:** GNN (one message-passing layer with `ModGNNConv`).

**Alternatives:**
- **MLP** on a flattened observation: not permutation-invariant.
  Renumbering robots would be a different input. The policy would
  have to learn to be invariant from data, which is a huge amount of
  sample inefficiency.
- **Transformer** (self-attention over robots): would also be
  permutation-invariant, more expressive than a single GNN layer, but
  ~10× more parameters and slower inference. Overkill for 4 robots.

**Tradeoff:** the GNN's communication is restricted to `comm_range`
neighbours. With only 4 robots in a 35 cm formation that's never a
problem, but for larger swarms you'd need more layers (each layer
extends communication by one hop).

### 8.3 Beta vs Gaussian action distribution

**Chose:** Beta(α, β) on each action dimension, squashed to
`[-MAX_V, MAX_V]`.

**Alternative:** Gaussian sampled then clipped, or `tanh`-squashed
Gaussian.

**Why Beta wins:**
- Native bounded support — no clipping artefacts at the boundaries.
- `α, β > 1` (enforced via softplus + 1) keeps it unimodal.
- Empirically more stable than Gaussian + clip on bounded continuous
  control tasks (Chou et al. 2017).

**Tradeoff:** the `(α, β) > 1` constraint can make the Beta
distribution overly smooth and prevent crisp deterministic actions.
In practice the entropy term decays naturally as training progresses.

### 8.4 Fixed-length rollouts vs episodic batches

**Chose:** fixed `T = 600` step rollouts, episodes can span
iteration boundaries.

**Alternative:** collect exactly N complete episodes per iteration.

**Why fixed-T wins:** PPO batch sizes need to be predictable for
optimizer stability and clip-ratio dynamics. Variable-length episodic
batches mean updates of wildly different sizes, which destabilises
training. Fixed-T side-steps that. The only complication is GAE
needs to handle partial episodes at the rollout boundary — which is
exactly what the `nextnonterminal` machinery in §6.3 does.

### 8.5 Hungarian assignment vs fixed-index slots

**Chose:** Hungarian-assign active robots to formation slots each
step.

**Alternative:** assign by robot index (robot 0 → slot 0, robot 1 →
slot 1, …).

**Why Hungarian wins:** the GNN is permutation-invariant. If the
reward assigned slots by index, the policy would have to learn the
specific mapping "robot index → slot", which fights the architectural
inductive bias and is utterly arbitrary anyway. With Hungarian, the
reward only cares about the *set* of robot positions vs the *set* of
slot positions — same invariance as the policy.

**Tradeoff:** O(n³) per step. For n=4 it's microseconds; for n=20 it
would matter.

### 8.6 Keep teleop'd robots in the world vs remove them

**Chose:** the teleop'd robot stays physically present and visible
to the policy, but its action is overridden and its reward is zero.

**Alternative:** remove it entirely from `pos`, `vel`, etc. — make
the cluster appear to be exactly `n_active` robots.

**Why ours wins:** the teleop'd robot is a real obstacle to be
avoided. Removing it from the policy's observation would mean the
policy might try to walk through it. Keeping it in the GNN graph
also means information flows: the policy sees the teleop'd robot's
position and can route around it.

**Tradeoff:** the policy must learn that `teleop_mask == 1` means
"avoid but don't try to formation with this one". The
`use_masks=True` features give it exactly this signal.

### 8.7 CSV/JSONL metrics vs tensorboard

**Chose:** plain text CSV/JSONL, optional matplotlib plot.

**Alternative:** tensorboard, wandb.

**Why ours wins for this project:** simple to inspect with `cat`,
`jq`, or `pandas`. No viewer dependency, no upload step, diff-friendly
across runs. JSONL grows new fields without breaking historical files.

**Tradeoff:** no live training dashboard. For a longer / larger
project I'd add tensorboard back; for this prototype the simpler
format won.

### 8.8 Curriculum (no-teleop phase 1, then teleop) vs single phase

**Recommended in README, but the actual successful run used
single-phase.**

**Chose:** single phase with teleop on from iter 1, 4648 iterations.

**Why it worked:** the GNN policy is biased enough toward formation
behaviour that teleop disturbance doesn't catastrophically prevent
early learning. The disturbance is also rare (`p_grab = 0.005` →
expected one grab per 200 steps per env), so early in training most
steps are clean and look like the no-disturbance task anyway.

**Tradeoff:** marginally slower to converge than a curriculum would
have been. But simpler to run and reason about.

### 8.9 `MAX_AGENTS = 4` fixed buffer + masking vs variable-length tensors

**Chose:** fixed `MAX_AGENTS = 4` everywhere, vary "active count"
through the mask.

**Alternative:** ragged tensors / dynamic batch dimensions.

**Why ours wins:** PyTorch's tensor primitives all assume fixed
shapes. Dynamic shapes would force every minibatch to be sized
independently — devastating for GPU throughput and PPO update
stability. The mask-based approach has a tiny constant overhead
(extra zero entries) for huge simplicity gains.

**Tradeoff:** doesn't generalise to genuinely variable team sizes
beyond `MAX_AGENTS`. For "what if a 5th robot joins?" you'd have to
re-train. For our problem statement (always 4 robots, 0–3 of them
teleop'd) it's perfect.

---

## 9. Empirical results

Numbers below are from `runs/20260503_195621_hallway-v1/`, which
trained from scratch for 4648 PPO iterations on a single MacBook Pro
CPU over ~19 hours. The full settings are pinned in the run's
`config.json`:

```
--iterations 5000 --num-envs 16 --max-steps 600
--checkpoint-every 200 --seed 0
PPO: γ=0.995, λ=0.95, clip=0.2, lr=5e-5, num_sgd_iter=8
RandomTeleop: p_grab=0.005, p_release=0.01, drift_speed=0.6
```

(The user interrupted at iter 4648, before the planned 5000.)

### 9.1 Headline numbers (after eval on 50 episodes per regime)

| Regime | Success rate | Mean reward | Episode length (sec) | Mean fwd velocity | Formation error |
|---|---|---|---|---|---|
| Clean (no teleop) | **100%** (50/50) | +244.90 | 13.9 s | 0.72 m/s | **1.6 cm** |
| Disturbed (RandomTeleop on) | **100%** (50/50) | +187.84 | 16.9 s | 0.61 m/s | 3.1 cm overall, 8.4 cm in 3-active phases, 2.3 cm in 4-active phases |

For context: agent radius is 8 cm, formation slot spacing is 35 cm.
A formation error of 1.6 cm means each robot is **within 5% of its
target slot** — visually indistinguishable from the perfect square.

### 9.2 Learning curve (bucketed from `episodes.jsonl`)

```
iter_bucket  episodes  success%   mean_R   mean_form_err   mean_collisions
       0       4786     89.6%    -18.24    0.068           7.88
     200       6007     99.7%   +140.15    0.045           3.72
     400       6165     99.6%   +147.50    0.041           3.63
     600       6224     99.8%   +169.02    0.039           2.01
     800       6419     99.9%   +180.13    0.035           2.25  ← peak
    1000       6526     99.5%   +176.41    0.037           1.43
    1200       6434     98.8%   +163.96    0.038           1.88
    1400       6004     97.6%   +153.89    0.038           1.64
    ...
    2200       4671     96.1%    +89.76    0.051           2.65  ← wobble #1
    ...
    3800       3991     73.4%    -27.21    0.063           3.80  ← wobble #2 (entropy spike?)
    4000       4358     63.1%    +15.16    0.051           3.10
    4200       5748     98.7%   +131.19    0.046           2.09  ← recovered
    4400       5328     97.9%   +126.00    0.043           2.39
    4600       1385     99.6%   +174.85    0.034           1.23  ← strong basin, latest.pt
```

**Three takeaways:**

1. **Strong early learning.** Success rate climbs from 89.6% to 99.9%
   in the first 800 iters; mean episode reward goes from -18 to +180.
   Formation error halves (0.07 → 0.035). Collisions drop 5×.
2. **Two PPO wobbles** at iters ~2200 and ~3800. Both are classic
   PPO instability signatures: too-large a policy update temporarily
   degrades the value function, the policy chases the bad advantage
   estimates, and learning regresses for ~200 iters before
   stabilising. Both recovered. The wobbles correlate with `approx_kl`
   spikes and `entropy` collapses in the CSV — a lower learning rate
   or higher entropy coefficient would have damped them.
3. **`latest.pt` (iter 4600) is in a strong basin.** Saving the
   latest checkpoint was the right call this time; some runs you'd
   instead want the *best-on-eval* checkpoint, which is a future
   improvement.

### 9.3 Per-active-count breakdown (disturbed eval)

```
formation_err per N : {'3': 0.0845, '4': 0.0233}
```

The policy holds 4-robot formation extremely tightly (2.3 cm) and
3-robot formation acceptably (8.4 cm). The 8.4 cm is partly because
the triangle requires more re-arrangement than the square (every
robot may have to move when one drops out), and partly because
3-active phases are a smaller fraction of total time — the policy
has seen them less.

Notably absent from this breakdown: keys `'2'` and `'1'`. **The
policy has never seen 2-active or 1-active configurations during
training**, because RandomTeleop only grabs one robot at a time. See
§10.

---

## 10. Known limitations

Listed roughly in order of presentation impact (most likely to be
asked about first).

### 10.1 The 2-robot line and 1-robot solo were never trained on

`RandomTeleop` is hardcoded to maintain at most one active grab per
env. So during 4600 iterations of training, the policy saw cluster
configurations with `n_active ∈ {3, 4}` only — never 2 or 1. The
target-formation-positions for n=1 and n=2 are correctly defined in
the env, but the policy has zero training signal for those regimes.

**What this means for the demo:** "press 1 to teleop one robot →
cluster forms triangle" works (verified, formation_err 8 cm). "Press
1 then 2 → cluster forms line of two" depends on whether the policy
generalises out-of-distribution. The GNN architecture *could* enable
this (it's permutation-invariant and naturally handles different
neighbour counts), but there's no guarantee.

**Fix (not yet implemented):** in `RandomTeleop`, allow multiple
concurrent grabs. ~10 lines of change to `code/teleop.py`. Then resume
training for 1000–2000 iters from `latest.pt` and the policy should
learn 2-active and 1-active too.

### 10.2 CPU-only training; never tested on GPU

The whole pipeline runs on CPU. It works (~19 hours for 4648 iters
at num_envs=16) but I never moved it to GPU. The model and rollouts
are both small enough that GPU would only be a 2–4× speedup, but
worth doing for longer sweeps. Would require changing `device="cpu"`
to `"cuda"` in `make_config()` plus a one-line change in the env to
move tensors to the GPU device.

### 10.3 No formation rotation

The target formation is always axis-aligned (square's sides parallel
to x/y, triangle's apex pointing +y). In a real setting you might
want the formation to rotate to face the direction of travel. For a
straight hallway this doesn't matter; for curved environments it
would.

### 10.4 No obstacle generalisation

The hallway is empty. The original AFOR baseline handles obstacles
(staggered walls). If the goal were "dynamic formation in cluttered
environments", we'd need to either (a) re-introduce obstacles in
`env_hallway.py` or (b) keep the obstacle-aware reward terms from
`env_line.py`. Out of scope for this iteration.

### 10.5 PPO instability at iters ~2200 and ~3800

Both wobbles recovered, but they're real and would have looked bad if
the run had been killed mid-wobble. Mitigation:

- Lower learning rate (5e-5 → 3e-5) — more stable, slower convergence.
- Higher entropy coefficient (0.001 → 0.01) — prevents premature
  policy collapse.
- Implement `--save-best-on-eval` so we keep the strongest checkpoint
  rather than just `latest.pt`.

### 10.6 The headline `mean_reward` in iterations.csv is a cumulative ratio

`code/train_hallway.py` computes `mean_reward = total_reward / global_step`
where both quantities are cumulative across all iterations. So the
"mean_reward" column in the CSV decays toward 0 over time even as
per-iteration rewards improve. The truthful learning curve comes from
the bucketed analysis of `episodes.jsonl` (shown in §9.2), not the
headline column. Worth fixing in a follow-up.

---

## 11. Glossary

One-liners. Useful as flashcards.

| Term | Definition |
|---|---|
| **MDP** | Markov Decision Process — formal RL setting (S, A, P, R, γ). |
| **Policy** | A (possibly stochastic) mapping from states to actions. |
| **Return** | Discounted sum of future rewards. |
| **Value function V(s)** | Expected return starting from state s under the current policy. |
| **Q-function Q(s, a)** | Expected return after taking action a in state s, then following the policy. |
| **Advantage A(s, a)** | Q(s, a) − V(s) — how much better than average this action is. |
| **PPO** | Proximal Policy Optimization — clipped policy gradient with multiple SGD epochs per batch. |
| **Clip ratio ε** | Upper bound on |πnew/πold − 1| in PPO; we use 0.2. |
| **GAE** | Generalised Advantage Estimation — λ-weighted blend of n-step TD advantages. |
| **CTDE** | Centralised Training, Decentralised Execution — multi-agent paradigm. |
| **GNN** | Graph Neural Network — node features updated by aggregating neighbour messages. |
| **Permutation invariance** | Output unchanged when inputs are reordered. |
| **Beta distribution** | Continuous distribution on [0, 1] parametrised by α, β > 0. |
| **Hungarian algorithm** | O(n³) optimal bipartite assignment. |
| **Rollout** | A fixed-length window of (s, a, r, s′) tuples collected before a PPO update. |
| **Iteration** | One PPO cycle: rollout + advantage computation + multiple SGD epochs. |
| **Episode** | One run of the env from reset to done; can span iteration boundaries. |
| **Formation error** | Mean distance between active robots and their Hungarian-assigned slots. |
| **Teleop mask** | Per-robot binary feature; 1 means the robot is under external (human / synthetic) control. |
| **Active cluster** | Robots with `teleop_mask == 0`; their count selects the target shape. |
| **`MAX_AGENTS`** | Fixed buffer width = 4 in our project. |

---

## 12. Likely audience questions and model answers

Use these as preparation. I've tried to anticipate the harder ones.

**Q1 — Why PPO instead of SAC, TD3, or anything more recent?**

PPO is the standard for on-policy multi-agent control because it's
simple, stable in practice, and parallelises trivially across env
copies. SAC and TD3 are off-policy actor-critic methods better suited
to single-agent continuous control with replay buffers; for our CTDE
multi-robot setting, PPO's batch-then-update structure maps more
cleanly. Also — the AFOR baseline used PPO, so we kept the same
algorithm to make the comparison apples-to-apples.

**Q2 — Why a GNN if you only have 4 robots? Couldn't an MLP work?**

It could, but it would have to learn permutation invariance from
data — a huge amount of sample inefficiency. The GNN bakes that
invariance in architecturally. It also generalises naturally if you
later want to scale to larger swarms.

**Q3 — Why Hungarian assignment in the reward instead of letting the policy learn slot assignment?**

Same answer pattern: the reward signal should match the architectural
inductive bias of the policy. Both are permutation-invariant. If the
reward used fixed-index assignment, the policy would have to expend
capacity learning a meaningless mapping, fighting the GNN's
invariance.

**Q4 — How does the policy know the cluster size has changed?**

Through the `teleop_mask` feature in the per-robot input. The active
cluster size is `sum(1 - teleop_mask)`, which the policy sees
implicitly via the masks of all 4 robots. The GNN aggregates neighbour
features which include those masks, so the message-passing computes
something equivalent to "how many active neighbours do I have?"

**Q5 — How is loss masking different from just zeroing the teleop'd robots' rewards?**

Two layers of safety. Zeroing rewards (which the env does) prevents
the *value function* from being trained on bogus targets. Masking the
loss (which the trainer does) prevents the *policy gradient* from
flowing through actions that weren't actually taken by the policy
(they were the human's). You need both — the value function is also
trained from the same loss, so masking handles both pathways.

**Q6 — Why use a fixed buffer width of 4 instead of variable-length tensors?**

PyTorch primitives expect fixed shapes. Variable-length batches kill
GPU throughput and destabilise PPO's clip-ratio dynamics. The mask
approach has microscopic overhead for huge simplicity gains.

**Q7 — What happens if all 4 robots are teleop'd at once?**

Then the active-mask sum is 0, which is clamped to 1 in the
normalisation. No gradient flows through the policy that step (zero
contribution to the loss). The env still steps, the teleop'd robots
still move, but training is paused for that env-step. This never
happens with `RandomTeleop` (single-grab limit) but can happen during
the demo if you press 1, 2, 3, and 4 in sequence.

**Q8 — Why dt = 0.05? Why max_v = 1.0?**

`dt = 0.05` (20 Hz) is a standard control rate for top-down 2D
robotics envs — fast enough that velocity changes feel responsive,
slow enough that PPO sees enough state change per step to learn from.
`max_v = 1.0` m/s is borrowed from the AFOR baseline; it makes the
hallway traversal time at full speed = 10 sec, so an episode of 600
steps (= 30 sec) gives the cluster three "tries" worth of time to
get to the goal.

**Q9 — Your CSV says `mean_reward` is going down over iterations. Is that bad?**

That column is misleadingly defined as `total_reward / global_step`
where both are *cumulative* across iterations — so it asymptotically
decays as global_step grows, even when per-iteration rewards are
improving. The real learning curve is in `episodes.jsonl` analysed
in §9.2, which shows mean episode reward climbing from -18 to +180
over the first 800 iters and stabilising in the +130 to +180 range
thereafter. This is a known minor reporting bug (§10.6).

**Q10 — Why `MAX_AGENTS = 4` and not, say, 6 or 8?**

The problem statement is "exactly 4 robots, some of which may be
under teleop". 4 was the smallest interesting number that gives all
four target shapes (square, triangle, line, solo). For 6 or 8 we'd
need to define more shapes (hexagon? wedge? half-line + half-circle?)
and re-train, but the architecture would handle it without changes.

**Q11 — How long would training take on a GPU?**

Estimated 4–6 hours for the same 4648 iterations (vs 19 hours on
CPU). The bottleneck is the rollout phase, not the update — and the
env runs in PyTorch tensors so it does benefit from GPU. Not
verified.

**Q12 — Could this transfer to real robots?**

Sim-to-real is its own research project. The action space is 2D
desired velocity, so a real diff-drive or omnidirectional robot
could consume the policy output. The big gaps are: (1) noise and
delay in real sensors / actuators, which we don't model; (2) the
policy was trained in a small bounded world (2 m × 12 m), which
might not generalise to larger spaces; (3) we have no hardware-level
collision recovery. A reasonable next step would be domain
randomisation in sim and then a small amount of fine-tuning on a
real robot platform.

**Q13 — What's the wall-clock cost of one PPO iteration?**

At `num_envs=16, max_steps=600, num_sgd_iter=8`, each iteration is
~14 seconds on the CPU — 9.6k env-steps for the rollout plus 8 ×
600 = 4800 minibatch updates. Most of the time is the rollout (env
step + policy forward pass).

**Q14 — Why does the policy slow down to 0.72 m/s when max is 1.0?**

The cluster trades forward speed for formation tightness. Going at
full speed makes formation harder to maintain — robots overshoot
their target slots. At 0.72 m/s the policy has found a sweet spot
where the formation reward and forward reward are jointly maximised.
You could push it faster by lowering `k_form` relative to `k_fwd`,
but at the cost of looser formation.

**Q15 — What single change would most improve the project?**

Letting `RandomTeleop` grab multiple robots concurrently, then
resume-training the existing checkpoint for 1000–2000 more iters.
That's the only thing standing between us and "demonstrably handles
2-robot line and 1-robot solo regimes". A 10-line code change and a
~3-hour training run.

---

*End of study guide. If you remember nothing else: **single shared
GNN policy, Beta-distribution actions, PPO with per-robot loss
masking by `(1 - teleop_mask)`, Hungarian-assigned formation reward,
100% success rate after 4600 iterations**.*

# Plan — Dynamic Formation PPO with Teleop-in-the-Loop

> **Note**: Plan mode restricts edits to this file. Per `CLAUDE.md` line 20, the implementation pass should also drop a copy of this plan at `/Users/aneeshsathe/Desktop/afor/plan.md` as the first step after approval.

---

## 1. Context

The repo currently ships three near-identical PPO environments — `code/env_line.py`, `code/env_pentagon.py`, `code/env_wedge.py` — each hardcoded to **5 robots** and a **single formation**. The existing trainer (`code/train.py`) and Beta-policy GNN model (`code/model.py`) are likewise pinned to `n_agents = 5`. The pipeline does work end-to-end (vectorised pygame envs + PPO on a GNN that already does permutation-invariant message passing within a comm-radius), but it cannot:

1. Spawn a configurable number of robots and switch the *target formation* based on the active cluster size.
2. Accept a "this robot is being driven by a human" override mid-episode and recover the formation around the loss/return of a teammate.
3. Train robustly to that disturbance pattern.

The goal is to extend AFOR (paper: arxiv 2404.01618) to **dynamic-size formation control with human teleop in the loop**:

- 4 robots spawn at one end of a long obstacle-free hallway.
- Cluster target shape switches based on number of policy-controlled robots: **4 → square**, **3 → triangle**, **2 → horizontal line**, **1 → solo** (degenerate case, just drive to goal).
- During inference, a human can grab/release any robot at any time (keyboard).
- During training, a synthetic "fake teleop" disturbance simulates this: pick a robot, drive it on an off-cluster trajectory for a stretch, then return it.
- A single shared PPO policy handles all cluster sizes via masking — the existing GNN is already permutation-invariant, so this is the minimal-code extension.
- Rewards: forward progress, formation maintenance, anti-collision, anti-stall.

The work is split across 4 people. **Decoupling is enforced by a single interface contract** (Section 3) that everyone codes against from day 1. After that contract is locked, all 4 streams proceed in parallel and only re-converge at integration.

---

## 2. High-Level Approach

A single new env module — `code/env_hallway.py` — replaces (does not modify) the three existing env files. It adopts the same `PassageEnv`/`PassageEnvRender` skeleton from `env_line.py` but generalises:

- `MAX_AGENTS = 4` (fixed buffer width); active count varies per step via masks.
- Long rectangular hallway, no obstacles. World ≈ `2.0m × 12.0m`. Spawn near `y = -5`, goal at `y ≈ +5`.
- Two orthogonal masks per robot, both in the observation:
  - `present_mask[i] ∈ {0,1}` — is robot `i` part of the simulation at all (always 1 in v1; reserved for future "robot-out-of-world" cases).
  - `teleop_mask[i] ∈ {0,1}` — is robot `i` currently under human / synthetic-teleop control.
- The **active cluster** = robots with `teleop_mask == 0`. The target formation is selected by `active_count`:
  - 4 → square (side `s`), 3 → equilateral triangle (side `s`), 2 → horizontal line (gap `s`), 1 → no formation term.
- Teleoperated robots **physically remain in the world** (so policy must avoid colliding with them) but are excluded from the formation reward.
- Reusable from existing code (do not re-derive): `compute_agent_dists`, `compute_obstacle_dists`, kinematic integration with velocity clipping + acceleration limits, the pygame coordinate transform, the rollout buffer scaffolding in `train.py`.

The PPO model gains a tiny extension: the per-robot input vector grows by 2 (one bit each for `teleop_mask` and `present_mask`), and the loss is masked so gradients flow only through policy-controlled robots. The number of agents passed through the GNN stays fixed at `MAX_AGENTS = 4` — teleoperated robots are kept in the graph as obstacles (the GNN sees their state but their action is overridden and their loss contribution is zeroed).

---

## 3. Interface Contract (lock this on Day 1, do not change unilaterally)

This is the ONE thing all four people agree on before splitting. Put it in `code/contract.py` as a docstring + dataclass; everyone imports `MAX_AGENTS`, `WORLD_DIM`, etc., from there.

```python
# code/contract.py
MAX_AGENTS = 4
DT = 0.05
WORLD_W = 2.0
WORLD_H = 12.0
SPAWN_Y = -5.0
GOAL_Y  = +5.0
MAX_V   = 1.0
MAX_A   = 2.0
AGENT_RADIUS = 0.08
FORMATION_SCALE = 0.35  # side / gap of square/triangle/line in metres
```

`FormationHallwayEnv(gym.Env)` (lives in `code/env_hallway.py`):

| Member | Type / shape | Notes |
|---|---|---|
| `observation_space` | `Dict` | keys: `pos (4,2)`, `vel (4,2)`, `goal (2,)`, `teleop_mask (4,)`, `present_mask (4,)`, `time (1,)` |
| `action_space` | `Box(low=-MAX_V, high=MAX_V, shape=(4,2))` | full action always supplied; env overrides teleop slots |
| `reset(seed=None)` | → `obs, info` | |
| `step(action)` | → `obs, reward (4,), done, trunc, info` | per-robot reward; loss masking handled by trainer |
| `set_teleop(idx:int, active:bool)` | side-effect | flips `teleop_mask[idx]`. Idempotent. |
| `set_teleop_action(idx:int, vel:np.ndarray)` | side-effect | sets the override velocity for the next `step`. |
| `target_formation_positions(active_count)` | → `(active_count, 2)` | pure function, exposed for visualisation |
| `render(mode='human'\|'rgb_array')` | → `None` or `ndarray` | reuses `PassageEnvRender` pattern |

**Rule:** anything not in this table is implementation-private. If you need a new field, post in the team channel before adding it — the masks/buffer shape ripple through everyone's code.

---

## 4. Reward Design (single source of truth)

Per-robot reward for robot `i` with `teleop_mask[i] == 0`:

| Term | Formula | Coeff |
|---|---|---|
| Forward progress | `+ k_fwd * (y_i_t - y_i_{t-1})` | `k_fwd = 5.0` |
| Stall penalty | `- k_stall` if cluster centroid moved < `eps` over last `K` steps | `k_stall = 0.5`, `K = 20`, `eps = 0.02` |
| Formation error | `- k_form * mean( |dist_to_target_slot| )` after Hungarian-assigning robots to slots of the target shape | `k_form = 2.0` |
| Inter-robot collision | `- k_coll` if any pairwise dist < `2 * AGENT_RADIUS` | `k_coll = 5.0` |
| Wall collision | `- k_wall * abs(x_overshoot)` | `k_wall = 1.0` |
| Goal bonus | `+ k_goal` once `mean(y) > GOAL_Y` for the active cluster | `k_goal = 20.0` |

For robot `i` with `teleop_mask[i] == 1`: reward is `0` (will be masked out anyway). Coeffs are starting points; expect to tune.

The Hungarian assignment is per-step over the `active_count` policy-controlled robots vs the `active_count` slots from `target_formation_positions(active_count)`, centred on the cluster centroid and rotated to align long-axis with `+y`. Use `scipy.optimize.linear_sum_assignment` (already in deps).

---

## 4.5 Metrics & Logging (persisted, for cross-run comparison)

Every training run writes to a per-run directory `runs/<timestamp>_<tag>/` (e.g. `runs/20260502_193000_hallway-v0/`). Three files are emitted, all append-only so a crashed run still yields a partial record:

**`config.json`** (one-shot, written at run start)
- All PPO hyperparameters (gamma, lambda, clip, lr, epochs, batch sizes, num_envs, num_iterations)
- Reward coefficients from Section 4 (`k_fwd`, `k_stall`, `k_form`, `k_coll`, `k_wall`, `k_goal`)
- Env constants from `contract.py`
- `RandomTeleop` parameters (`p_grab`, `p_release`, `drift_speed`)
- Git commit SHA, timestamp, machine name, torch + CUDA versions

**`iterations.csv`** (one row per PPO iteration — this is the primary comparison artefact)
Columns: `iter, wall_time_s, env_steps, policy_loss, value_loss, entropy, total_loss, approx_kl, clip_frac, grad_norm, mean_reward, mean_episode_length, lr`. Write after the SGD update for each iteration.

**`episodes.jsonl`** (one JSON object per finished episode — richer per-event detail)
Fields: `iter, env_id, episode_length, total_reward, reached_goal (bool), num_collisions, num_wall_hits, num_teleop_grabs, max_active_count, min_active_count, formation_error_mean, formation_error_per_active_count (dict {2,3,4 → float}), forward_velocity_mean, stall_steps`. Logged when `done` fires.

**Checkpoints** continue to land at `runs/<run>/weights/weights_epoch{i}.pt` (keep every 50th iter + last). A `weights/latest.pt` symlink points at the most recent.

A tiny `code/scripts/compare_runs.py` reads N run dirs and prints a table + saves `runs/_comparison/<timestamp>.png` plots (mean reward + formation error per active-count, over iterations). This is a stretch deliverable — the data is the must-have, plots are nice-to-have.

**Why these formats:** CSV for metrics that are tabular and compared across runs (pandas-friendly, opens in any tool). JSONL for richer per-episode events that may grow new fields without breaking historical files. JSON for the one-shot config so it's diffable.

---

## 5. Work Split (4 people, minimal blocking)

All four start by reading Section 3 and confirming the contract via PR. After that, the dependency DAG is essentially flat — each person has stubs they can develop against.

### Person 1 — Env Core (`code/env_hallway.py`)

**Deliverable:** `FormationHallwayEnv` and `FormationHallwayEnvRender` (a thin pygame wrapper) implementing the contract in Section 3.

**Tasks**
1. Copy the `PassageEnv` skeleton from `env_line.py` (lines 39–426) into `env_hallway.py`. Keep `compute_agent_dists`, `sample_pos_noise`, the kinematic integrator, the wall-clip logic. Drop all obstacle code.
2. Set world to `WORLD_W × WORLD_H`. Spawn 4 robots in a small jittered square around `(0, SPAWN_Y)`. Goal `(0, GOAL_Y)`.
3. Implement `target_formation_positions(n)` for `n ∈ {1,2,3,4}`. Pure function; unit-test it standalone.
4. Implement reward terms from Section 4. Hungarian assignment with `scipy.optimize.linear_sum_assignment`.
5. Implement `set_teleop` / `set_teleop_action`. In `step`, overwrite slots of the incoming `action` array where `teleop_mask == 1` with the stored override velocities (default zero if none set).
6. Vectorised version (`num_envs > 1`) — keep parity with the existing `PassageEnv` pattern; needed for fast PPO rollouts. If this turns out costly, ship a single-env version first and let Person 3 wrap with `gym.vector` as a stopgap.

**Independent deliverables / tests**
- `pytest tests/test_formation.py` — formation positions are correct shape, centred at origin, expected distances.
- `python -m code.env_hallway --random` — spins up the env with random actions and prints rewards (no rendering needed for this).

**Depends on:** Section 3 contract only.

---

### Person 2 — Teleop (`code/teleop.py`)

**Deliverable:** Two classes that drive the env's teleop interface.

**Tasks**
1. `RandomTeleop(env, p_grab=0.005, p_release=0.01, drift_speed=0.6)` — at each step, with prob `p_grab` if any robot is currently free, pick one and call `env.set_teleop(i, True)`; with prob `p_release` if any teleop'd, free it. While teleop'd, push that robot on a sinusoidal lateral drift away from the cluster centroid for a randomly-sampled duration. **This is what runs during training.**
2. `KeyboardTeleop(env)` — pygame key handler:
   - `1/2/3/4` → toggle teleop on robot 1..4
   - `WASD` → drive currently-selected teleop robot
   - `0` → release all
   Returns the per-step velocity vector to pass to `env.set_teleop_action`. **This is what runs during eval.**
3. Demo script `python -m code.teleop --demo` that opens a tiny pygame window with mocked positions and prints which robot is teleop'd. Lets Person 2 develop without waiting for Person 1.

**Independent deliverables / tests**
- Stub env (`tests/fake_env.py`) implements just `set_teleop` / `set_teleop_action` / `step` returning fixed obs. Person 2 unit-tests both teleop classes against this.

**Depends on:** Section 3 contract only.

---

### Person 3 — Model + Trainer (`code/model.py`, `code/train.py`)

**Deliverable:** PPO trains stably on `FormationHallwayEnv` with the random-teleop disturbance active.

**Tasks**
1. **Model patch (`model.py`):**
   - Change `self.n_agents = 5` (line 99) → read from env, expect `4`.
   - Grow per-robot input feature dim from 6 → 8 (append `teleop_mask[i]`, `present_mask[i]`). Update encoder input dim at line 45.
   - Output / Beta-distribution path is unchanged.
2. **Trainer patch (`train.py`):**
   - Swap `from env_line import PassageEnv` for `from env_hallway import FormationHallwayEnv`.
   - Pre-allocate buffers with `MAX_AGENTS = 4`.
   - Wire in `RandomTeleop` from Person 2: each env owns one instance, called inside the rollout loop just before `env.vector_step`.
   - **Loss masking:** in the PPO update, multiply per-robot policy and value losses by `(1 - teleop_mask)` before reducing. Renormalise by `sum(1 - teleop_mask)` instead of `n_agents`.
   - **Persistent metrics (Section 4.5):** at the start of training, create `runs/<timestamp>_<tag>/`, write `config.json`. After each PPO iteration, append a row to `iterations.csv` with all the columns listed in 4.5 (read them out of the PPO update — `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `clip_frac`, `grad_norm` are all already computed locally; just plumb them into a writer). On every `done` flag inside the rollout loop, append a JSON line to `episodes.jsonl` with the per-episode aggregates (track these in a small per-env accumulator that resets on `reset()`). Keep one `print` per iteration as a convenience but make the CSV/JSONL the source of truth.
   - Checkpoint to `runs/<run>/weights/weights_epoch{i}.pt`; maintain `weights/latest.pt` symlink.
3. PPO hyperparameters: keep current values (gamma 0.995, lambda 0.95, clip 0.2, lr 5e-5, 10 SGD epochs, train_batch 65536, minibatch 4096). Drop `num_envs` from 32 → 16 if memory tight on 4 robots.

**Independent deliverables / tests**
- Develop against a stub `FakeHallwayEnv` (≈30 lines, returns random obs of the right shape) until Person 1's env is ready. The stub satisfies the Section 3 contract.
- Sanity run: 100 iterations should show non-decreasing mean reward.

**Depends on:** Section 3 contract; uses `RandomTeleop` from Person 2 at integration time.

---

### Person 4 — Render + Eval + Integration glue (`code/render_hallway.py`, `code/eval.py`, `code/run_demo.py`)

**Deliverable:** A working `python -m code.run_demo --weights weights/hallway-v0/weights_epoch_latest.pt` that opens a pygame window, runs the trained policy, and lets the user keyboard-teleop any robot.

**Tasks**
1. `render_hallway.py` — extend the existing `PassageEnvRender` pattern: draw 4 colored circles, target-formation outline (faded), arrow from each robot to its formation slot, a visible "TELEOP" marker on currently-teleop'd robots, hallway walls, goal line.
2. Update `eval.py`: load checkpoint with the new model architecture (input dim 8 instead of 6), step env at `1/DT` Hz, print episode reward.
3. `run_demo.py`: wires `FormationHallwayEnv` + `KeyboardTeleop` (Person 2) + trained policy (Person 3) + renderer. This is the demo binary.
4. Develop the renderer against Person 1's env stub or hand-rolled fake state — colored circles in a hallway.

**Independent deliverables / tests**
- `python -m code.render_hallway --self-test` cycles through N=4/3/2/1 with hard-coded positions and verifies all four formation overlays draw correctly.

**Depends on:** Section 3 contract; uses real env from Person 1 and trained weights from Person 3 only at the final demo.

---

## 6. Integration Milestones

| Day | Milestone | Owner |
|---|---|---|
| 0 | `contract.py` merged. All 4 people can `from contract import *`. | Whoever opens the repo first |
| 2 | Stubs ready: Person 3 has `FakeHallwayEnv`, Person 2 has `tests/fake_env.py`, Person 4 has fake-state renderer. **Critical: nobody is blocked.** | All |
| 4 | Person 1's env passes its self-test (random actions, sensible rewards). | P1 |
| 5 | Person 3 swaps stub → real env, kicks off first real training run. | P3 |
| 5 | Person 4 swaps fake state → real env render. | P4 |
| 6 | Person 2's `KeyboardTeleop` integrated into Person 4's demo. | P2 + P4 |
| 7 | First end-to-end demo: trained policy, human pulls a robot out, formation collapses 4→3, robot returns, formation re-forms. | All |

---

## 7. Critical Files & Reuse Map

**New files (do not touch existing envs):**
- `code/contract.py` — constants + interface docstring
- `code/env_hallway.py` — env (Person 1)
- `code/teleop.py` — `RandomTeleop` + `KeyboardTeleop` (Person 2)
- `code/render_hallway.py` — pygame renderer (Person 4)
- `code/run_demo.py` — interactive demo (Person 4)
- `code/metrics.py` — tiny `RunLogger` class wrapping `config.json` / `iterations.csv` / `episodes.jsonl` writers (Person 3, ~80 lines)
- `code/scripts/compare_runs.py` — read N run dirs, print + plot comparisons (Person 3, stretch)
- `tests/test_formation.py`, `tests/fake_env.py` — fixtures
- `runs/` — created on first training run, .gitignored

**Files modified in place:**
- `code/model.py` — small patches at lines 45, 99, 119 to make `n_agents` and input dim parametric (Person 3)
- `code/train.py` — env import, buffer dims, teleop integration, masked loss (Person 3)
- `code/eval.py` — load new architecture, attach renderer (Person 4)

**Files left alone:**
- `code/env_line.py`, `code/env_pentagon.py`, `code/env_wedge.py` — keep working as the published baseline.

**Reusable functions to lift verbatim** (cite by `file:line`):
- `code/env_line.py:115` `sample_pos_noise`
- `code/env_line.py:123` `compute_agent_dists`
- `code/env_line.py:270-286` velocity-clip + acceleration-limit + position-update kinematics
- `code/env_line.py:436-491` pygame coordinate transform + draw primitives
- `code/model.py:11-30` `ModGNNConv` (no changes)
- `code/train.py:121-187` rollout buffer + GAE (only the `n_agents` dim changes)

---

## 8. Verification

### Per-component smoke tests (each owner runs locally before merge)
- **Env:** `pytest tests/test_formation.py` and `python -m code.env_hallway --random --steps 200` — no crashes, rewards in sane range, formation reward ≈ 0 when robots placed exactly on slots.
- **Teleop:** `python -m code.teleop --demo` — pygame window shows teleop selection cycling.
- **Model/Trainer:** Sanity training run of 100 iterations on 4-robot env with random teleop active — mean reward should increase, no NaNs in loss.
- **Renderer:** `python -m code.render_hallway --self-test` — visual confirmation of all 4 formation overlays.

### End-to-end acceptance test
1. Train: `python -m code.train --iterations 5000 --tag hallway-v0` (≈1–2 hrs on a single GPU; tune iterations down for the smoke run). Confirm `runs/<ts>_hallway-v0/iterations.csv` is growing and `episodes.jsonl` has rows for both teleop'd and non-teleop'd episodes.
2. Eval headless: `python -m code.eval --weights runs/<ts>_hallway-v0/weights/weights_epoch_latest.pt --episodes 20 --no-render` — also writes `runs/<ts>_hallway-v0/eval.json` (per-episode reward, cluster forward velocity, formation error per active count, success bool). Pass criterion: cluster reaches goal in ≥80% of episodes when no teleop disturbance is applied at eval.
2b. Sanity-check the metrics: `python -m code.scripts.compare_runs runs/<ts>_hallway-v0` — confirm policy_loss trends down, mean_reward trends up, formation error per active-count is finite for all of {2,3,4}.
3. Demo: `python -m code.run_demo --weights weights/hallway-v0/weights_epoch_latest.pt`. Manually verify:
   - Cluster moves down hallway in a square.
   - Press `1` → robot 1 highlighted as teleop, drive it sideways with WASD; remaining 3 should re-form a triangle.
   - Press `1` again to release; robot 1 should rejoin and the cluster should re-form a square.
   - Pull two robots out → remaining 2 form a horizontal line.
4. Stretch: record a 30-sec video of the demo for the README.

### What is explicitly out of scope (so we don't get stuck polishing)
- No tensorboard / wandb (print logging is fine for the prototype).
- No curriculum learning over teleop probability — fixed `p_grab` is enough.
- No formation rotation to arbitrary headings — long-axis aligned to `+y` (direction of travel) is enough for a hallway.
- No multi-policy ensemble per cluster size — single shared policy is the target.
- No obstacle generalisation — empty hallway only.

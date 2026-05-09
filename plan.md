# Plan — Always-circle formation with dynamic robot count (1..10)

## 1. Context

We are abandoning the per-active-count multi-formation policy (4→square, 3→triangle, 2→line) shipped on `aneesh/policy_v2`. The new objective:

- The cluster **always** forms a **circle** and moves across the arena toward the goal line.
- The number of robots is **dynamic at runtime**, between **1 and 10**.
- Through teleop the user can: (a) drive any robot manually, (b) spawn a new robot, (c) delete an existing robot.
- Training simulates these disturbances with a `RandomTeleop` that randomly grabs / releases / spawns / deletes.
- Arena geometry changes: **8 m × 8 m square** (no longer a long thin hallway). Robot radius bumped to **0.2 m**.

This branch is `aneesh/circle_policy_v1`, branched off the clean `aneesh/policy_v2` head. Code that doesn't apply anymore is deleted (not commented out). Everything is trained from scratch — v2/v3 weights won't load (action space size differs).

## 2. Architecture decision

**Buffers stay sized to `MAX_AGENTS=10`. `present_mask` becomes the live source of truth for "does this robot exist right now."** This re-uses the same masking pattern the env already has for `teleop_mask`, avoids any dynamic tensor reshaping, and keeps rollout buffers at a fixed `(T, num_envs, 10, 2)` shape end-to-end.

A robot's runtime state is two bits:

| `present_mask` | `teleop_mask` | meaning |
|---:|---:|---|
| 1 | 0 | **active** — in the world, GNN/policy controls it, counts toward the circle, earns rewards |
| 1 | 1 | **teleop'd** — in the world, human-controlled, does **not** count toward the circle, earns no policy reward |
| 0 | 0 | **deleted** — not in the world; not rendered; excluded from GNN, formation, rewards, collisions |
| 0 | 1 | invalid — `delete()` always clears teleop too |

Active cluster size: `n_active = sum(present_mask * (1 - teleop_mask))`. The circle target is built for `n_active` robots.

Non-present robots are parked at a far-away sentinel position so `radius_graph` naturally drops them from message passing, and their action outputs are masked to zero in the loss.

## 3. Critical files

| file | role | what changes |
|---|---|---|
| `code/contract.py` | constants + REWARD_COEFFS | `MAX_AGENTS=10`, add `MIN_AGENTS=1`, `INITIAL_AGENTS=4`, `AGENT_RADIUS=0.2` (was 0.08), `WORLD_W=WORLD_H=8.0` (was 2×12), `SPAWN_Y=-3.5`, `GOAL_Y=3.5`, formation now circle-only — drop `FORMATION_SCALE` for `CIRCLE_SIDE=0.6` (must exceed `2·AGENT_RADIUS=0.4` plus margin) |
| `code/env_hallway.py` | env, formation reward, step | rewrite `target_formation_positions` for circle; gate every reward by `present_mask`; add `spawn(env_idx, robot_idx=None)` / `delete(env_idx, robot_idx)`; rewrite `get_starts_and_goals` to spawn only the initial set; non-present robots parked at `(SENTINEL_X, SENTINEL_Y)` outside `WORLD_H`; goal check uses active-cluster centroid; episode terminates if `n_present == 0` |
| `code/model.py` | GNN + Beta policy | already reads `n_agents` from obs space — flips to 10 cleanly. `use_masks=True` already wired by all callers. Sentinel positions handle GNN edge filtering — no model code change needed. |
| `code/teleop.py` | KeyboardTeleop + RandomTeleop | KeyboardTeleop: 1-9 toggle teleop on robot 1-9, `0` toggles robot 10, `=`/`+` spawns, `-`/`_` deletes the most-recently-selected robot, `r` releases all, WASD/Z/X unchanged. RandomTeleop: add `p_spawn` / `p_delete` and `init_n_present_dist`; ensure 1 ≤ `n_present` ≤ 10 always. |
| `code/render_hallway.py` | pygame renderer | Iterate `present_mask` for robot drawing. Formation overlay draws a circle of `n_active` slots at variable radius. HUD shows `n_present / n_active / n_teleop`. Square arena geometry. Robot color palette extended to 10. |
| `code/eval_hallway.py` | per-regime eval | Replace `--fixed-active-count` with `--fixed-n-present` (1..10). Bucket episodes by `min_n_present` seen during episode; per-regime stats over k ∈ {1..10}. |
| `code/train_hallway.py` | PPO trainer | Buffer alloc uses `MAX_AGENTS=10`. Loss masked by `present_mask * (1 - teleop_mask)`. Drop the per-active-count CSV columns; replace with `mean_n_present`, `mean_formation_error`, `mean_circle_radius`. CLI flags `--p-spawn` / `--p-delete` / `--init-n-present-dist`. |
| `code/metrics.py` | CSV schema | Drop `formation_error_active_{1..4}` and `n_episodes_active_{1..4}`. Add `mean_n_present`, `mean_formation_error`, `mean_circle_radius`. `EpisodeAccumulator` tracks `min_n_present`. |
| `code/run_demo.py` | demo binary | Inherits new KeyboardTeleop keys; refresh help text and HUD line. |
| `tests/test_formation.py` | unit tests | Delete square/triangle/line/`unsupported_count_raises`. Add: `test_circle_n_points`, `test_circle_centred`, `test_circle_radius_scaling`, `test_circle_n1_returns_origin`. |
| `tests/test_teleop.py` | teleop tests | Drop `init_regime_dist`-of-length-4 assumptions. Add `test_spawn_increments_present_count`, `test_delete_decrements_present_count`, `test_delete_clears_teleop`, `test_min_one_robot_invariant`, `test_max_ten_robots_invariant`. |

## 4. Circle geometry

`target_formation_positions(n_active)` returns `n_active` points on a circle centred at origin. Spacing: aim for a constant inter-neighbour chord length `CIRCLE_SIDE = 0.6`. This must exceed `2·AGENT_RADIUS = 0.4` so adjacent robots don't permanently overlap; 0.6 leaves a 0.2 m centre-to-centre margin. Then `r(n) = CIRCLE_SIDE / (2 sin(π/n))`. Special cases: `n=1` → `[[0,0]]`; `n=2` → `[±CIRCLE_SIDE/2, 0]`.

| n  | r       | diameter | outer extent (incl. robot bodies) |
|----|---------|----------|------------------------------------|
| 1  | —       | 0        | 0.4                                |
| 2  | 0.30    | 0.60     | 1.00                               |
| 3  | 0.35    | 0.69     | 1.09                               |
| 4  | 0.42    | 0.85     | 1.25                               |
| 5  | 0.51    | 1.02     | 1.42                               |
| 6  | 0.60    | 1.20     | 1.60                               |
| 7  | 0.69    | 1.38     | 1.78                               |
| 8  | 0.78    | 1.57     | 1.97                               |
| 9  | 0.88    | 1.75     | 2.15                               |
| 10 | 0.97    | 1.94     | 2.34                               |

Even at n=10 the cluster footprint (2.34 m) fits comfortably inside the 8 m × 8 m arena. Hungarian assignment from active robots → slots — same code path that exists today, just over `n_active`.

## 5. Reward surface (single global set, no per-count keys)

Per-step, per-robot, summed:

- `+k_fwd · dy` — forward progress in y (gated by `present * (1-teleop)`).
- `-k_form · slot_dist` — Hungarian distance to circle slot (gated by `present * (1-teleop)`).
- `-k_coll` — pairwise collision penalty among present robots only.
- `-k_wall · overshoot_x` — x-boundary overshoot (present robots only).
- `-k_stall` — applied if the active centroid hasn't moved for `stall_window` steps.
- `+k_goal` — one-shot when the active centroid reaches `GOAL_Y`.

Coefficients start at the v2 defaults (`k_fwd=5, k_stall=0.5, k_form=2, k_coll=5, k_wall=1, k_goal=20`). Tune after the first smoke run.

Episode termination: `timeout` OR `goal_reached` OR `n_present == 0` (degenerate empty cluster).

## 6. Spawn / delete semantics

- **Spawn:** picks the lowest free index where `present_mask[i] == 0`. Position: cluster centroid + small random offset; velocity zero; teleop_mask cleared. No-op if all 10 slots occupied.
- **Delete:** clears `present_mask[i]`, `teleop_mask[i]`, `teleop_vels[i]`; parks position at sentinel `(WORLD_H*2, WORLD_H*2)` so it falls outside `radius_graph` and is never drawn. No-op if `n_present == 1` (preserve min-1 invariant).
- KeyboardTeleop `delete` removes the **currently-selected** (last-toggled-on-teleop) robot if there is one, else the highest-index present robot.

## 7. Random disturbance schedule (training)

`RandomTeleop` runs four independent Bernoulli per step per env:
- `p_grab`: grab a present, non-teleop'd robot for a random duration with sinusoidal lateral push.
- `p_release`: release a teleop'd robot.
- `p_spawn` (NEW): spawn a new robot if `n_present < 10`. Default 0.002.
- `p_delete` (NEW): delete a present, non-teleop'd robot if `n_present > 1`. Default 0.002.

At episode reset, sample initial `n_present ∈ [1..10]` from a coverage-flat distribution. Pre-seed `present_mask` accordingly.

All p-values tunable from the trainer CLI: `--p-grab`, `--p-release`, `--p-spawn`, `--p-delete`, `--init-n-present-dist`.

## 8. 4-person task split (parallel-friendly)

**Step 0 (~30 min, blocks everyone):** Person A lands the API surface — contract.py constants and env_hallway.py method **signatures** for `spawn`, `delete`, and `target_formation_positions(n) → circle`, with stub bodies that just toggle the masks (no reward changes yet). Push to `aneesh/circle_policy_v1` so B/C/D can branch off.

After step 0, B/C/D run independently:

| Person | Files | Deliverable | Blocked by |
|---|---|---|---|
| **A: env + reward** | `contract.py`, `env_hallway.py` | Real circle target, present-mask-gated rewards, sentinel positions for non-present, goal check uses active centroid, env smoke green | self |
| **B: training + metrics** | `train_hallway.py`, `metrics.py` | Buffers sized to MAX_AGENTS=10, loss masked by present, dropped per-active CSV columns, replaced with present-aware columns; trainer launches and writes valid CSV/JSON | A's stubs |
| **C: teleop + renderer** | `teleop.py`, `render_hallway.py` | KeyboardTeleop with 0-9 / `=` / `-` / `r`, RandomTeleop with p_spawn / p_delete and `init_n_present_dist`, renderer iterates present_mask and draws circle outline + updated HUD | A's stubs |
| **D: eval + tests + demo** | `eval_hallway.py`, `tests/test_formation.py`, `tests/test_teleop.py`, `run_demo.py` | Eval with `--fixed-n-present`, tests rewritten for circle, demo with refreshed help text and key bindings | A's stubs |

Final integration (~1 h, after A/B/C/D converge): A merges, runs the trainer smoke (50 iters fixed n=4) + tests, then runs trainer smoke (50 iters dynamic n) and confirms PPO health metrics.

## 9. Verification (in order, each gates the next)

1. `pytest tests/` — all pass.
2. `python code/env_hallway.py --random --steps 200` — env smoke runs without crash.
3. `python code/teleop.py --demo` — RandomTeleop visits all `n_present` values 1..10 in 400 steps with default disturbance probabilities.
4. **PPO health smoke** — 50 iters from scratch, `n_present` fixed at 4 (`--p-spawn 0 --p-delete 0 --init-n-present-dist 0,0,0,1,0,0,0,0,0,0`):
   - `entropy` rises (less negative each iter)
   - `approx_kl` ≥ 1e-4 by iter 20
   - `clip_frac` ≥ 0.02 by iter 50
   - score improves
5. **Dynamic-count smoke** — 100 iters with default disturbance schedule. `mean_n_present` over the run is roughly uniform in [1..10] and the trainer doesn't crash.
6. **Full run** — 800 iters, `--save-best-on-eval --eval-every 25 --eval-episodes 10 --eval-n-present-counts 1,2,5,10`. Target per-regime success ≥ 60% in each evaluated count.
7. **Visual demo** — `python code/run_demo.py --weights runs/<ts>_circle-v1/weights/best.pt --reset-on-done`. Walk-through:
   - 4 robots spawn, form a circle, move toward goal.
   - Press `=` three times → 7 robots, circle expands smoothly.
   - Press `-` twice → 5 robots, circle contracts smoothly.
   - Press `3` to teleop robot 3 away with WASD; the remaining cluster reforms a circle and continues forward.
   - Verify min-1 / max-10 invariants by spamming `=` and `-`.

## 10. Out of scope

- Warm-starting from v2/v3 weights (action_space size changes 4→10).
- Heterogeneous robots / per-robot capabilities.
- Obstacles in the arena.
- Non-circle formation shapes — explicitly removed.
- 3D, more than 10 robots, multiple cooperating clusters.
- MPS/GPU — CPU only.

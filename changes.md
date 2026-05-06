# Changes made

## Performance improvements

1. **Automatic teleoperation probability schedule**
   - Added a teleop curriculum in `code/train_hallway.py` that linearly increases `RandomTeleop.p_grab` from `p_grab` to `p_grab_final` over training iterations.

2. **Cluster-conditional formation weighting**
   - Updated `code/env_hallway.py` formation reward so `k_form` scales by active cluster ratio `k / n_agents`, reducing over-penalization when fewer policy-controlled robots are active.

3. **Radius curriculum**
   - Added environment-level radius curriculum fields: `agent_radius_start`, `agent_radius_end`, `agent_radius_curriculum_steps`.
   - Added runtime progression with `set_curriculum_step()` and `current_agent_radius()`.
   - Collision and wall constraints now use `current_agent_radius()`.

## CUDA port

4. **CUDA-first training/eval device selection**
   - Added `--device` CLI option to `code/train_hallway.py` and `code/eval_hallway.py`.
   - Default behavior prefers CUDA when available.
   - Environment now receives selected device through config.

## End-to-end training/eval pipeline

5. **Single-command pipeline script**
   - Added `scripts_run_pipeline.sh`.
   - Script runs: training -> evaluation -> run comparison.
   - Prints locations of `config.json`, `iterations.csv`, `episodes.jsonl`, and `eval.json` for immediate inspection.

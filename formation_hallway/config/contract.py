"""
Single source of truth for the env contract.

Mirrors multi-robot-formation/code/contract.py exactly. If you ever change
those constants there, change them here too, or import the original via
PYTHONPATH. Kept as a local copy so this package is self-contained.
"""

# robots
MAX_AGENTS = 4

# time + world
DT = 0.05
WORLD_W = 2.0
WORLD_H = 12.0

# spawn / goal lines (along world y axis; long hallway dimension)
SPAWN_Y = -5.0
GOAL_Y = 5.0

# kinematics
MAX_V = 1.0      # per-axis velocity bound, m/s
MAX_A = 2.0      # acceleration bound, m/s^2
MIN_A = -2.0

# robot geometry
AGENT_RADIUS = 0.08      # pygame sim radius. Real Triton chassis is 0.10 m.

# formation slot scale (square is +/- FORMATION_SCALE/2 corner offset)
FORMATION_SCALE = 0.35

# episode caps
DEFAULT_MAX_TIME_STEPS = 600

# reward coeffs (kept here so the env node can compute eval metrics if desired)
REWARD_COEFFS = {
    "k_fwd": 5.0,
    "k_stall": 0.5,
    "k_form": 2.0,
    "k_coll": 5.0,
    "k_wall": 1.0,
    "k_goal": 20.0,
    "stall_window": 20,
    "stall_eps": 0.02,
}

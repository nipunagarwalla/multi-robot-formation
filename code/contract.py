"""Interface contract for the dynamic-formation hallway env.

Everything that touches FormationHallwayEnv (env code, teleop, trainer,
renderer, eval, demo) imports from here. Changing a constant ripples
through everyone's code, so do it deliberately.

Observation dict (per env):
    pos          : list[[x, y]] of length MAX_AGENTS
    vel          : list[[vx, vy]] of length MAX_AGENTS
    goal         : list[[gx, gy]] of length MAX_AGENTS  (broadcast for compat)
    teleop_mask  : list[float] of length MAX_AGENTS in {0., 1.}
    present_mask : list[float] of length MAX_AGENTS in {0., 1.}
    time         : list[[t]] of length MAX_AGENTS  (broadcast for compat)

Action: numpy array shape (num_envs, MAX_AGENTS, 2) of desired velocities.
The env overrides slots where teleop_mask == 1 with the stored teleop velocities.

Active cluster size = sum(present_mask * (1 - teleop_mask)) per env.
Target formation: 4->square, 3->triangle, 2->horizontal line, 1->no formation term.
"""

MAX_AGENTS = 4

DT = 0.05
WORLD_W = 2.0
WORLD_H = 12.0

SPAWN_Y = -5.0
GOAL_Y = 5.0

MAX_V = 1.0
MAX_A = 2.0
MIN_A = -2.0

AGENT_RADIUS = 0.08

FORMATION_SCALE = 0.35

DEFAULT_MAX_TIME_STEPS = 600
DEFAULT_RENDER_PX_PER_M = 60

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

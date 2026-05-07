# formation_hallway

A ROS1 / Gazebo package that deploys a trained multi-robot PPO policy on
4 Triton holonomic robots navigating a 2 × 12 m hallway in formation.
The Gazebo environment is built to mirror the pygame training environment
exactly, so a trained checkpoint can be loaded and run without any changes.

## Overview

The policy is trained in a lightweight pygame simulation
([multi-robot-formation](https://github.com/aneesh-sathe/multi-robot-formation))
where 4 robots learn to move down a hallway while maintaining a dynamic
formation that adapts to how many robots are under policy control. This
package provides the Gazebo side: the world, robot models, velocity plugin,
and a policy runner node that speaks the same observation/action contract as
the training environment.

## Package layout

```
formation_hallway/
├── plugins/model_push.cc        ← Gazebo plugin: body-frame Twist → world-frame velocity
├── models/triton/               ← Triton robot SDF + mesh
├── worlds/hallway.world         ← 2 × 12 m walled hallway with spawn and goal lines
├── launch/hallway.launch        ← Gazebo + 4-robot spawner
├── scripts/
│   ├── spawn_hallway_fleet.py   ← spawns 4 robots in a jittered square near the spawn line
│   ├── teleop_hallway.py        ← toggle multi-robot teleop (keys 1–4 + WASD)
│   └── hallway_env_node.py      ← loads policy weights and drives robots at 20 Hz
├── config/contract.py           ← shared constants (world size, spawn/goal lines, kinematics)
└── policy/                      ← place your latest.pt here
```

## Prerequisites

- ROS Noetic + Gazebo 11
- A catkin workspace (`~/catkin_ws`)
- Python 3.9+ with the packages from `multi-robot-formation/requirements.txt`
  installed (torch, torch_geometric, gymnasium, numpy, etc.)
- A trained checkpoint (`latest.pt`) from the `multi-robot-formation` repo

## Build

```bash
ln -s /path/to/formation_hallway ~/catkin_ws/src/
cd ~/catkin_ws && catkin_make
source devel/setup.bash
```

## Quickstart

Three terminals, all with `source ~/catkin_ws/devel/setup.bash`:

```bash
# Terminal 1 — bring up the world and spawn 4 robots
roslaunch formation_hallway hallway.launch

# Terminal 2 — (optional) manual teleop
#   1–4: toggle teleop on robot 1–4   WASD: drive selected robot
#   0: release all   ESC: quit
rosrun formation_hallway teleop_hallway.py

# Terminal 3 — run the policy (use your python3.9+ venv, not rosrun)
source ~/your_venv/bin/activate
python3 ~catkin_ws/src/formation_hallway/scripts/hallway_env_node.py \
    --weights /path/to/multi-robot-formation/runs/<timestamp>/weights/latest.pt \
    --policy-repo /path/to/multi-robot-formation
```

> **Why `python3` directly instead of `rosrun`?**  
> `rosrun` uses the system Python, which may not have torch installed.
> Running the script directly with your virtualenv's Python bypasses the
> catkin wrapper and picks up all installed packages correctly. ROS topics
> still work as long as `ROS_MASTER_URI` is set (which sourcing `setup.bash` handles).

## How env parity with the training sim is maintained

| Concern | What this package does |
|---|---|
| World size | 2 × 12 m (walls at `x = ±1`, end caps at `y = ±6`) |
| Long axis | `y` is the hallway axis, same as pygame |
| Spawn / goal | `SPAWN_Y = −5`, `GOAL_Y = +5`; centroid-based goal check |
| Step rate | 20 Hz via `rospy.Rate(20)` |
| State source | reads directly from `/gazebo/model_states` — no odom node needed |
| Obs dict | `pos / vel / goal / teleop_mask / present_mask / time`, same shapes as training |
| Velocity frame | `/gazebo/model_states` twist is world-frame; matches what the policy expects |
| Action units | per-robot `[vx, vy]` in m/s, world-frame, clipped to `[−1, +1]` |
| Frame conversion | env node rotates world → body before publishing Twist; the plugin rotates back |
| Yaw stabilisation | proportional damping `angular.z = −k_yaw × yaw` keeps body ≈ world frame |
| Teleop override | teleop node publishes cmd_vel for masked robots; policy skips those indices |
| Episode reset | `/gazebo/set_model_state` teleports robots to jittered spawn slots, yaw=0 |

## Policy node flags

```
python3 hallway_env_node.py --weights <path> --policy-repo <path> [options]
```

| Flag | Default | Purpose |
|---|---|---|
| `--weights` | required | path to a `.pt` checkpoint |
| `--policy-repo` | required | path to the `multi-robot-formation` directory |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--max-steps` | 600 | episode length cap before auto-reset |
| `--k-yaw` | 1.0 | yaw damping gain; increase if robots drift rotationally |
| `--no-autoreset` | off | stop after the first episode (useful for debugging) |

## Known differences from the training sim

| | Training (pygame) | Gazebo (this package) |
|---|---|---|
| Robot radius | 0.08 m | 0.10 m (real Triton chassis) |
| Formation slot clearance | 0.19 m | 0.15 m |
| Physics | kinematic (instant velocity) | Gazebo rigid body with the velocity plugin |

The tighter clearance (0.15 m) is fine in practice. If you observe
consistent jostling within the formation, set `AGENT_RADIUS = 0.10` in
`multi-robot-formation/code/contract.py` and retrain.

## What this package does not do

- **Train.** Training runs entirely in pygame against `FormationHallwayEnv`.
- **Compute reward.** The policy node only checks the goal condition for
  episode resets. Reward shaping lives in the training environment.
- **Use lidar.** The Triton SDF includes a lidar mast and publishes `/scan`,
  but the policy ignores it — observations are position and velocity only.

# limo_circle_sim

ROS2 Humble + Gazebo Classic deployment of the `circle_policy_v1` multi-robot
formation PPO policy on a fleet of AgileX LIMO robots.

This package mirrors the ROS1 reference (`formation_hallway/` on the
`nipun/gazebo` branch, originally targeting Triton robots). The policy and
constants are imported directly from `multi-robot-formation/code/` — no
abstraction layer.

## What it does

- Spawns `max_agents` (default 10) LIMOs in Gazebo. The first `num_agents`
  (default 6) sit on a circle around `(0, SPAWN_Y=-3.5)`; the rest are
  parked at `(SENTINEL=24, 24)` off the visible arena.
- A single `circle_node` reads all robot poses from `/model_states` (one
  topic, all robots at once — no per-robot odom subs), runs the trained
  PPO policy at 20 Hz, and publishes `/limo_<i>/cmd_vel` (Twist).
- The policy emits **world-frame** `(vx, vy)` per the contract in
  `code/env_hallway.py`; the node rotates that to body frame plus a yaw
  damper so the LIMO's stock diff-drive plugin can execute it.
- A `teleop_node` opens a pygame window. Keys (same as
  `code/teleop.py:KeyboardTeleop`):

  | key | action |
  |---|---|
  | `1`-`9`, `0` | toggle teleop on robot 1-10 |
  | `W A S D` | drive selected robot (world frame: w=+y, s=-y, a=-x, d=+x) |
  | `Z` / `X` | decrease / increase drive speed |
  | `=` / `+` | spawn (flip a free slot's present_mask to 1) |
  | `-` / `_` | delete selected/last robot (flip present_mask to 0) |
  | `R` | release all teleop'd robots |
  | `ESC` / `Q` | quit |

  Spawn / delete are realized by `circle_node` teleporting the LIMO
  model between sentinel and the active centroid via `/set_entity_state`.
  No model creation/destruction at runtime.

## Build

```bash
cd /path/to/multi-robot-formation     # repo root, branch nipun/limo-gazebo
source /opt/ros/humble/setup.bash
colcon build --symlink-install --base-paths ros2/limo_circle_sim
source install/setup.bash
```

The package's Python module imports the policy code via `sys.path` — make
sure `multi-robot-formation/code/` is intact and that `torch`,
`gymnasium`, `numpy`, `scipy`, `pygame` are importable in your active
Python environment.

## Run

The launch only brings up Gazebo + the LIMO fleet by default. Run the
policy and teleop nodes yourself in separate terminals — easier to
restart, swap checkpoints, or attach a debugger.

**Terminal 1 — Gazebo + fleet:**
```bash
ros2 launch limo_circle_sim circle_sim.launch.py num_agents:=6
```

**Terminal 2 — policy node:**
```bash
ros2 run limo_circle_sim circle_node --ros-args \
    -p weights:=$HOME/multi-robot-formation/weights/latest.pt \
    -p num_agents:=6
```

**Terminal 3 — keyboard teleop:**
```bash
ros2 run limo_circle_sim teleop_node --ros-args -p num_agents:=6
```

If you'd rather have everything come up from one launch (less
flexibility, but one window), pass the opt-in flags:

```bash
ros2 launch limo_circle_sim circle_sim.launch.py \
    weights:=$HOME/multi-robot-formation/weights/latest.pt \
    num_agents:=6 \
    use_policy:=true use_teleop:=true
```

Launch args:

| arg | default | notes |
|---|---|---|
| `num_agents` | `6` | initial present count, 1..max_agents |
| `max_agents` | `10` | must equal `code/contract.py:MAX_AGENTS` |
| `world` | `circle_arena.world` | |
| `use_policy` | `false` | if true, launch starts circle_node (needs `weights:=`) |
| `use_teleop` | `false` | if true, launch starts teleop_node |
| `weights` | `""` | absolute path to a `.pt` checkpoint; only required when `use_policy:=true` |

## Verifying the contract (smoke tests)

1. **Fleet smoke (no teleop).** 6 robots on a circle around `(0, -3.5)`
   advance toward `y = +3.5` while holding formation:

   ```bash
   ros2 launch limo_circle_sim circle_sim.launch.py \
       weights:=/abs/path/to/latest.pt use_teleop:=false
   ```

2. **World-frame velocity sanity** (single robot, manual override):

   ```bash
   ros2 launch limo_circle_sim circle_sim.launch.py \
       weights:=/abs/path/to/latest.pt num_agents:=1
   # in another shell:
   ros2 topic pub /teleop/mask    std_msgs/Float32MultiArray '{data: [1,0,0,0,0,0,0,0,0,0]}' -1
   ros2 topic pub /teleop/cmd     std_msgs/Float32MultiArray \
       '{data: [1.0,0.0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0]}' -r 20
   ```
   The first LIMO should drift toward world `+X` regardless of its yaw.
   Swap `[1.0,0.0,...]` for `[0.0,1.0,...]` to confirm `+Y` motion.

3. **Spawn / delete workflow** (the 7→9→5 sequence from the plan):

   ```bash
   ros2 launch limo_circle_sim circle_sim.launch.py \
       weights:=/abs/path/to/latest.pt num_agents:=7
   ```
   In the teleop window: press `+` twice (7 → 9), then press
   `1 -` `2 -` `3 -` `4 -` (toggle teleop on each, then delete; ends at 5).
   The formation should reform after each event.

## Known limitations

- **LIMO is diff-drive; the policy was trained as holonomic.** We rotate
  world→body and publish both `linear.x` and `linear.y`, but
  `gazebo_ros_diff_drive` only honors `linear.x`. Effect: pure lateral
  policy commands wait for the robot's yaw to drift before producing
  motion. Identical approximation to the ROS1 reference. Plan risk #1
  has the mitigation if needed (turn-toward-velocity heading controller).
- The depth camera is disabled in the URDF for fleet performance. The 2D
  LiDAR is still on.
- Real-robot bringup is out of scope here — no `/model_states` source on
  hardware. Follow-up plan.

## Layout

```
ros2/limo_circle_sim/
├── launch/circle_sim.launch.py     # single entrypoint
├── worlds/circle_arena.world       # 8x8 empty + lines + gazebo_ros_state
├── urdf/                           # vendored LIMO URDF (depth cam disabled)
├── meshes/                         # vendored LIMO meshes
└── limo_circle_sim/
    ├── circle_node.py              # policy wrapper, /model_states->/cmd_vel
    └── teleop_node.py              # pygame keyboard teleop
```

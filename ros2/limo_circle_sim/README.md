# limo_circle_sim

ROS2 Humble + Gazebo Classic deployment of two multi-robot formation PPO
policies on a fleet of AgileX LIMO robots:

- **`hallway_node`** *(default / showcase)* — runs the **`policy_v2`**
  checkpoint (`MAX_AGENTS = 4`, square / triangle / line / solo
  formation). Use this for the small-cluster reformation demo
  (4 → 3 → 2 → 3 → 4); this is the policy we recommend for new demos
  and what most of the snippets below assume.
- **`circle_node`** *(alternative)* — runs the **`circle_policy_v1`**
  checkpoint (`MAX_AGENTS = 10`, n-gon "circle" formation). Use this if
  you want 5–10 robots forming a ring and marching toward the goal.

Both wrappers share the same world, URDF, teleop, ground-truth TF, and
arena marker stack. The policy + checkpoint loader are imported directly
from `multi-robot-formation/code/` — no abstraction layer.

---

## Prerequisites

You must already have these working on the target machine.

- **Ubuntu 22.04** (Jammy)
- **ROS2 Humble**, sourced (`source /opt/ros/humble/setup.bash`)
- **Gazebo Classic 11**
- **`gazebo_ros_pkgs` for ROS2** (`gazebo_dev`, `gazebo_msgs`,
  `gazebo_plugins`, `gazebo_ros`).
- **`ros-humble-xacro`**, **`ros-humble-robot-state-publisher`**,
  **`ros-humble-joint-state-publisher`**, **`ros-humble-rviz2`**
- **`tf2_ros`**, **`tf2_tools`** (standard with Humble)
- **Python 3.10** (the Humble default) with these packages installed in
  the same Python ROS2 uses:
  ```
  torch                          # CPU build is fine
  torch-geometric
  torch-cluster                  # provides radius_graph
  gymnasium
  numpy
  scipy
  pygame
  ```

- A **trained `.pt` checkpoint** for whichever policy you want to run
  (v2 for `hallway_node`, v1 for `circle_node`).

---

## Build

```bash
cd /path/to/multi-robot-formation                  # repo root
source /opt/ros/humble/setup.bash
source /path/to/gazebo_ros_pkgs_ws/install/setup.bash   # if you source-built
ln -s "$PWD/ros2/limo_circle_sim" path/to/ros2_ws/src/limo_circle_sim
cd path/to/ros2_ws
colcon build --symlink-install --packages-select limo_circle_sim
source ~/circle_ws/install/setup.bash
```

`--symlink-install` means edits to the source files take effect without a
rebuild (except for `setup.py`/`package.xml` changes).

---

## Run

Three terminals. Source `path/to/ros2_ws/install/setup.bash` (+ ROS2 +
gazebo_ros_pkgs) in each. The launch starts Gazebo, spawns the fleet,
starts `gt_tf_node` and `markers_node`, and (optionally) starts rviz.
You then `ros2 run` the policy and teleop nodes in their own shells so
they can be restarted independently.

### Option A — `hallway_node` *(default, recommended)*

policy_v2, ≤ 4 robots, square / triangle / line / solo formations.

**T1 — Gazebo + fleet + rviz:**
```bash
ros2 launch limo_circle_sim circle_sim.launch.py \
    num_agents:=4 \
    total_robots:=4 \
    spawn_pattern:=hallway \
    headless:=true use_rviz:=true
```

Spawns 4 LIMOs in a 0.35 m square centered at `(0, -3.2)`. Add
`use_real_meshes:=true` to load the AgileX `.dae` visuals — see
[Visualization](#visualization) for when that's safe.

**T2 — policy:**
```bash
ros2 run limo_circle_sim hallway_node --ros-args \
    -p weights:=/abs/path/to/policy_v2.pt \
    -p num_agents:=4 \
    -p total_robots:=4
```

**T3 — teleop (note `max_agents:=4`):**
```bash
ros2 run limo_circle_sim teleop_node --ros-args \
    -p num_agents:=4 \
    -p max_agents:=4
```

Tested demo: moved robots out of the formation one at a time
(4 → 3 → 2) then back in (2 → 3 → 4). Cluster reforms after each
spawn / delete and keeps marching toward `+Y`.

### Option B — `circle_node` (alternative)

policy_v1, ≤ 10 robots, n-gon "circle" formation.

**T1 — Gazebo + fleet + rviz:**
```bash
ros2 launch limo_circle_sim circle_sim.launch.py \
    num_agents:=6 \
    total_robots:=6 \
    spawn_pattern:=circle \
    headless:=true use_rviz:=true
```

Spawns 6 LIMOs on a 0.6 m radius hexagon centered at `(0, -3.2)`. Each
robot starts at yaw = π/2 (facing the goal at `+Y`).

**T2 — policy:**
```bash
ros2 run limo_circle_sim circle_node --ros-args \
    -p weights:=/abs/path/to/circle_policy_v1.pt \
    -p num_agents:=6 \
    -p total_robots:=6
```

**T3 — keyboard teleop (optional):**
```bash
ros2 run limo_circle_sim teleop_node --ros-args \
    -p num_agents:=6 \
    -p max_agents:=10
```

Tested demo: `num_agents:=7 total_robots:=9` then `+ + + - - - -` to
walk the cluster 7 → 9 → 5 active.

---

## Visualization

The launch starts both `gzserver` (physics) and rviz2 by default. There
are two distinct visualizers and two distinct visual representations
for the LIMOs — pick the combination that works on your machine.

### Visualizers

| Visualizer | Launch flag | Cost | Notes |
|---|---|---|---|
| **`gzclient`** (the Gazebo GUI) | `headless:=false` | heavy | photorealistic, OGRE-backed; needs real OpenGL acceleration |
| **`rviz2`** | `use_rviz:=true` | medium | preconfigured config under `config/circle_sim.rviz`; shows the LIMOs, walls, spawn / goal lines, and TF tree from a single fixed `world` frame |
| **headless** | `headless:=true use_rviz:=false` | lightest | gzserver runs but nothing renders — useful for batch runs / topic-only tests |

### LIMO appearance — primitives vs. real meshes

The URDF supports both styles, picked at URDF-load time by a launch arg:

| Launch arg | Visual | Per-robot data | Where it works |
|---|---|---|---|
| `use_real_meshes:=false` *(default)* | yellow box body + black cylinder wheels | ~1 KB | everywhere, including software-GL hosts (M1 UTM, headless CI) |
| `use_real_meshes:=true` | full AgileX `.dae` meshes (`limo_base.dae` + `limo_wheel.dae`) | ~63 MB | hosts with real GPU acceleration |

Just append it to whichever launch you're running:

```bash
ros2 launch limo_circle_sim circle_sim.launch.py \
    num_agents:=4 total_robots:=4 spawn_pattern:=hallway \
    headless:=false use_rviz:=true \
    use_real_meshes:=true
```

No rebuild needed — `--symlink-install` plus the xacro `<xacro:if
value="$(arg use_real_meshes)">` conditional means the choice is made
every time `robot_state_publisher` parses the URDF at launch.

Collision geometry is identical in both modes (box body, cylinder
wheels) so physics behavior doesn't change with this flag.


> **Why `use_real_meshes` defaults to `false`.** The package was
> developed on a MacBook Air (M1) running Ubuntu 22.04 inside UTM. UTM
> doesn't expose hardware OpenGL to Linux guests on Apple Silicon, so
> the only renderer available is Mesa's `llvmpipe` software rasterizer
> (confirmed via `glxinfo | grep "OpenGL renderer"` →
> `llvmpipe (LLVM 15.0.7, 128 bits)`). Software OpenGL trying to render
> the original AgileX `.dae` meshes — ~52 MB body + ~11 MB wheel ×
> 4 wheels per robot, multiplied by N robots in the fleet — caused
> `gzclient` and `rviz2` to be OOM-killed before they finished loading
> the URDF. Defaulting `use_real_meshes:=false` (box + cylinder
> primitives, ~1 KB per robot) lets viewers come up cleanly on any
> host. On a machine with a real GPU just pass
> `use_real_meshes:=true` and you get the full LIMO look — the
> physics, control, and policy code are independent of the visual
> representation.

---

## Launch arguments

| arg | default | notes |
|---|---|---|
| `num_agents` | `6` | initial present count |
| `total_robots` | `num_agents` | how many LIMO entities to actually spawn in Gazebo. Set higher than `num_agents` to park spares at the sentinel (24, 24) for teleop spawn/delete |
| `max_agents` | `10` | `MAX_AGENTS` from policy_v1's contract. Auto-clamped to 4 when `spawn_pattern:=hallway` |
| `spawn_pattern` | `circle` | `"circle"` = v1 n-gon (1..10 robots); `"hallway"` = v2 square/triangle/line (1..4) |
| `world` | `circle_arena.world` | the bundled 8×8 m world with walls, spawn/goal strips, and `gazebo_ros_state` |
| `use_rviz` | `false` | start rviz2 with the bundled config |
| `headless` | `false` | skip `gzclient` (use this on software-GL machines) |
| `use_real_meshes` | `false` | `true` to load the AgileX `.dae` mesh visuals; `false` for box+cylinder primitives. See [Visualization](#visualization) |
| `use_policy` | `false` | start `circle_node` from the launch — convenient one-window mode, requires `weights:=` |
| `use_teleop` | `false` | start `teleop_node` from the launch |
| `weights` | `""` | path to a `.pt` checkpoint; only required when `use_policy:=true` |
| `use_sim_time` | `true` | feed `/clock` into all our nodes |

---

## Topics published

| topic | type | producer | notes |
|---|---|---|---|
| `/model_states` | `gazebo_msgs/ModelStates` | `gazebo_ros_state` plugin in the world | ground-truth world pose of every entity; this is what the policy and `gt_tf_node` read |
| `/limo_<i>/cmd_vel` | `geometry_msgs/Twist` | `circle_node` or `hallway_node` | world-frame `(vx, vy)` rotated to the body frame the `planar_move` plugin expects |
| `/limo_<i>/odom` | `nav_msgs/Odometry` | `planar_move` plugin | encoder-style odometry (off by default for TF — see below) |
| `/tf` | `tf2_msgs/TFMessage` | `gt_tf_node` + per-robot `robot_state_publisher` | `world → limo_<i>/base_footprint` is broadcast directly from `/model_states` so rviz always reflects physics, never odometry drift |
| `/teleop/cmd` | `std_msgs/Float32MultiArray` | `teleop_node` | length `2 * max_agents` — `(vx, vy)` world-frame per slot |
| `/teleop/mask` | `std_msgs/Float32MultiArray` | `teleop_node` | length `max_agents` — which slots the policy should hand over to teleop |
| `/teleop/present` | `std_msgs/Float32MultiArray` | `teleop_node` | length `max_agents` — current present_mask, used by `circle_node`/`hallway_node` to teleport spares between sentinel and the active cluster |
| `/arena_markers` | `visualization_msgs/MarkerArray` | `markers_node` | walls + spawn/goal strips drawn into rviz |

---

## Teleop keyboard reference

`teleop_node` opens a small pygame status window. Click into it so it
captures keys, then:

| key | action |
|---|---|
| `1`-`9`, `0` | toggle teleop on robot 1..10 |
| `W A S D` | drive the selected teleop'd robot in world frame (W=+y, S=-y, A=-x, D=+x) |
| `Z` / `X` | decrease / increase teleop drive speed |
| `=` / `+` | spawn a new robot (flips next free slot to present) |
| `-` / `_` | delete the selected (or highest-index) robot |
| `R` | release all teleop'd robots |
| `ESC` / `Q` | quit |

`+` / `-` cause `circle_node` / `hallway_node` to call
`/set_entity_state` to teleport the LIMO model between the sentinel
position `(24, 24)` and the active formation centroid. No models are
actually created or destroyed at runtime.

---

## Cleaning up between runs

`gzserver` doesn't always release port 11345 cleanly if you `Ctrl-C`
mid-launch. Subsequent launches then fail with
`Service /spawn_entity unavailable. Was Gazebo started with
GazeboRosFactory?` Run this between recording takes:

```bash
pkill -9 -f "gzserver|gzclient|gazebo|spawn_entity|robot_state_publisher|joint_state_publisher|rviz2|markers_node|gt_tf_node|circle_node|hallway_node|teleop_node|static_transform_publisher" 2>/dev/null
rm -rf /tmp/.gazebo /tmp/gazebo* 2>/dev/null
sleep 1
```

---

## Known limitations

- **LIMO is physically diff-drive; the policies are holonomic.** We
  swap the URDF's `gazebo_ros_diff_drive` plugin for
  `gazebo_ros_planar_move`, which lets the body translate in any
  direction (effectively omni). This is the cheapest way to honor the
  policy's world-frame `(vx, vy)` actions without retraining. The
  wheels don't spin visually because `planar_move` is kinematic, but
  collisions with walls and other robots are still resolved by
  Gazebo's physics.
- The depth camera in the LIMO URDF is disabled. Fleet-wide depth
  streams would saturate the bus and stress the renderer. The 2D
  lidar is still on.
- Real-robot bringup is out of scope here — no `/model_states` source
  on hardware.

---

## Layout

```
ros2/limo_circle_sim/
├── package.xml
├── setup.py / setup.cfg
├── resource/limo_circle_sim
├── launch/
│   └── circle_sim.launch.py          # Gazebo + fleet spawn + rviz + helper nodes
├── worlds/
│   └── circle_arena.world            # 8x8 m arena, perimeter walls at ±4.3, spawn/goal strips, gazebo_ros_state plugin
├── meshes/                           # AgileX .dae meshes (loaded when use_real_meshes:=true)
├── urdf/                             # LIMO URDF; visual = primitive OR mesh, picked by use_real_meshes
│   ├── limo_four_diff.xacro
│   ├── limo_four_diff.gazebo         # gazebo_ros_planar_move plugin
│   ├── limo_xacro.xacro
│   └── limo_gazebo.gazebo
├── config/
│   └── circle_sim.rviz               # preconfigured rviz panel (Fixed Frame=world, 10 RobotModel slots, MarkerArray, Grid)
├── scripts/
│   └── make_fake_checkpoint.py       # creates a random-init .pt for plumbing tests
└── limo_circle_sim/
    ├── circle_node.py                # policy_v1 wrapper (n-gon, ≤10 robots)
    ├── hallway_node.py               # policy_v2 wrapper (square/triangle/line, ≤4 robots)
    ├── teleop_node.py                # pygame keyboard teleop
    ├── markers_node.py               # arena geometry → /arena_markers
    └── gt_tf_node.py                 # /model_states → world→<ns>/base_footprint TF
```
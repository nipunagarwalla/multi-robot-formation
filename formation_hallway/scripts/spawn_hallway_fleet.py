#!/usr/bin/python3
"""
Spawn N=4 Triton robots at the canonical square-formation slot offsets
around (0, SPAWN_Y), with small jitter, all yawed to 0.

Mirrors FormationHallwayEnv.get_starts_and_goals():
  - target_formation_positions(4) = square with vertices at +/- 0.175 m
  - each robot starts at SPAWN_Y + slot_y + uniform(-0.05, 0.05)
  - x is slot_x + uniform(-0.05, 0.05)

Usage:
  rosrun formation_hallway spawn_hallway_fleet.py
  rosrun formation_hallway spawn_hallway_fleet.py --n 4 --seed 0
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import rospy
import rospkg
from geometry_msgs.msg import Pose
from gazebo_msgs.srv import SpawnModel

# import the constants package next to us
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "config"))
from contract import (  # noqa: E402
    SPAWN_Y, FORMATION_SCALE, MAX_AGENTS,
)


def square_slots(scale: float):
    """Same layout as target_formation_positions(4) in env_hallway.py."""
    s = scale
    return [
        (-s / 2, -s / 2),
        ( s / 2, -s / 2),
        ( s / 2,  s / 2),
        (-s / 2,  s / 2),
    ]


def load_template():
    pkg = rospkg.RosPack().get_path("formation_hallway")
    with open(f"{pkg}/models/triton/model.sdf", "r") as f:
        return f.read()


# RGBA colors per robot index (0-based)
ROBOT_COLORS = [
    (1.00, 0.20, 0.20),  # robot 1 — red
    (0.20, 0.80, 0.20),  # robot 2 — green
    (0.20, 0.50, 1.00),  # robot 3 — blue
    (1.00, 0.85, 0.10),  # robot 4 — yellow
]


def _material_xml(r, g, b):
    return (
        f"<material>"
        f"<ambient>{r:.2f} {g:.2f} {b:.2f} 1</ambient>"
        f"<diffuse>{r:.2f} {g:.2f} {b:.2f} 1</diffuse>"
        f"</material>"
    )


def spawn_one(template, name, robot_idx, x, y, spawn_srv):
    sdf = template.replace("__NS__", name)
    # replace every existing <material>...</material> block with the robot's color
    color = ROBOT_COLORS[robot_idx % len(ROBOT_COLORS)]
    mat = _material_xml(*color)
    sdf = re.sub(r"<material>.*?</material>", mat, sdf, flags=re.DOTALL)
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = 0.0
    pose.orientation.w = 1.0  # yaw = 0
    spawn_srv(model_name=name, model_xml=sdf, robot_namespace=name,
              initial_pose=pose, reference_frame="world")
    rospy.loginfo(f"spawned {name} at ({x:+.3f}, {y:+.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=MAX_AGENTS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jitter", type=float, default=0.05,
                    help="uniform jitter half-range on each axis (m)")
    args = ap.parse_args(rospy.myargv()[1:])

    import numpy as np
    rng = np.random.default_rng(args.seed)

    rospy.init_node("spawn_hallway_fleet", anonymous=True)
    rospy.wait_for_service("/gazebo/spawn_sdf_model")
    spawn_srv = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
    template = load_template()

    slots = square_slots(FORMATION_SCALE)
    if args.n > len(slots):
        rospy.logwarn(f"n={args.n} > 4 slots; only 4 supported, capping")
        args.n = len(slots)

    for i in range(args.n):
        sx, sy = slots[i]
        jx = rng.uniform(-args.jitter, args.jitter)
        jy = rng.uniform(-args.jitter, args.jitter)
        x = sx + jx
        y = SPAWN_Y + sy + jy
        spawn_one(template, f"triton_{i+1}", i, x, y, spawn_srv)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        sys.exit(0)

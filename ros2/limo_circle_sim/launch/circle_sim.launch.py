"""
circle_sim.launch.py — single entry point for the circle_policy_v1 + LIMO
fleet Gazebo simulation.

Spawns `max_agents` LIMOs (default 10). The first `num_agents` are placed on
the target_formation_positions circle around (0, SPAWN_Y); the rest are
parked at the sentinel position (SENTINEL_X, SENTINEL_Y) ≈ (24, 24) m off
the visible arena, ready for the teleop node to flip into the formation
via the `+` key.

Then starts circle_node (policy) and optionally teleop_node.

Args:
  weights      (required)   absolute path to a .pt checkpoint
  num_agents   (default 6)  initial present count, 1..max_agents
  max_agents   (default 10) MUST equal code/contract.py:MAX_AGENTS
  world        (default circle_arena.world)
  use_teleop   (default true)
  use_sim_time (default true)

Usage:
  ros2 launch limo_circle_sim circle_sim.launch.py \\
      weights:=/abs/path/to/latest.pt \\
      num_agents:=6
"""
from __future__ import annotations

import os
import pathlib
import sys

import launch_ros.descriptions
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.substitutions import FindPackageShare


# Import contract constants and the formation slot function from code/.
# The launch file lives at multi-robot-formation/ros2/limo_circle_sim/launch/
# so three parents up is the repo root.
_HERE = pathlib.Path(__file__).resolve()
_REPO = _HERE.parents[3]
_CODE = _REPO / "code"
if not _CODE.is_dir():
    raise RuntimeError(f"could not find policy code at {_CODE}")
sys.path.insert(0, str(_CODE))

from contract import MAX_AGENTS, SPAWN_Y, SENTINEL_X, SENTINEL_Y  # noqa: E402
from env_hallway import target_formation_positions  # noqa: E402


URDF_REL = "urdf/limo_four_diff.xacro"
ENTITY_PREFIX = "limo_"


def _slot_poses(num_agents: int, max_agents: int):
    """Return list[(x, y, z, yaw)] of length max_agents.

    The first `num_agents` slots sit on the target_formation_positions
    circle, shifted to (0, SPAWN_Y). The rest park at the sentinel.
    """
    base = target_formation_positions(num_agents).cpu().numpy()  # (num_agents, 2)
    out: list[tuple[float, float, float, float]] = []
    for i in range(max_agents):
        if i < num_agents:
            x = float(base[i, 0])
            y = float(SPAWN_Y + base[i, 1])
        else:
            x = float(SENTINEL_X)
            y = float(SENTINEL_Y)
        out.append((x, y, 0.0, 0.0))
    return out


def _spawn_one(pkg_share, namespace: str, x: float, y: float, z: float, yaw: float,
               use_sim_time):
    urdf_path = PathJoinSubstitution([pkg_share, URDF_REL])
    robot_description_content = Command([
        "xacro", " ", urdf_path,
        " robot_namespace:=", namespace,
    ])
    return GroupAction([
        PushRosNamespace(namespace),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{
                "robot_description": launch_ros.descriptions.ParameterValue(
                    robot_description_content,
                    value_type=str,
                ),
                "use_sim_time": use_sim_time,
            }],
            remappings=[("/tf", "tf"), ("/tf_static", "tf_static")],
        ),
        Node(
            package="joint_state_publisher",
            executable="joint_state_publisher",
            parameters=[{"use_sim_time": use_sim_time}],
        ),
        Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            arguments=[
                "-entity", namespace,
                "-topic", "robot_description",
                "-x", str(x),
                "-y", str(y),
                "-z", str(z),
                "-Y", str(yaw),
                "-robot_namespace", namespace,
            ],
            output="screen",
        ),
    ])


def _build_fleet(context, *args, **kwargs):
    """Resolve runtime args and emit the per-robot spawn actions."""
    num_agents = int(LaunchConfiguration("num_agents").perform(context))
    max_agents = int(LaunchConfiguration("max_agents").perform(context))
    if max_agents != MAX_AGENTS:
        raise RuntimeError(
            f"max_agents={max_agents} != contract.MAX_AGENTS={MAX_AGENTS}; "
            "the policy buffers are sized for MAX_AGENTS."
        )
    if not (1 <= num_agents <= max_agents):
        raise RuntimeError(
            f"num_agents={num_agents} not in [1, {max_agents}]"
        )

    use_sim_time = LaunchConfiguration("use_sim_time")
    pkg_share = FindPackageShare("limo_circle_sim").find("limo_circle_sim")

    poses = _slot_poses(num_agents, max_agents)
    actions = []
    for i, (x, y, z, yaw) in enumerate(poses):
        ns = f"{ENTITY_PREFIX}{i + 1}"
        # stagger spawns to avoid spawn_entity racing on robot_description
        actions.append(
            TimerAction(
                period=0.4 * i,
                actions=[_spawn_one(pkg_share, ns, x, y, z, yaw, use_sim_time)],
            )
        )

    # circle_node — wait for all robots to be on the bus before starting
    actions.append(
        TimerAction(
            period=0.4 * max_agents + 2.0,
            actions=[
                Node(
                    package="limo_circle_sim",
                    executable="circle_node",
                    output="screen",
                    parameters=[{
                        "weights": LaunchConfiguration("weights"),
                        "num_agents": num_agents,
                        "max_agents": max_agents,
                        "entity_prefix": ENTITY_PREFIX,
                        "use_sim_time": use_sim_time,
                    }],
                ),
            ],
        )
    )

    actions.append(
        TimerAction(
            period=0.4 * max_agents + 2.0,
            actions=[
                Node(
                    package="limo_circle_sim",
                    executable="teleop_node",
                    output="screen",
                    parameters=[{
                        "num_agents": num_agents,
                        "use_sim_time": use_sim_time,
                    }],
                    condition=IfCondition(LaunchConfiguration("use_teleop")),
                ),
            ],
        )
    )

    return actions


def generate_launch_description():
    pkg_share = FindPackageShare("limo_circle_sim")
    default_world = PathJoinSubstitution([pkg_share, "worlds", "circle_arena.world"])
    pkg_gazebo_ros = FindPackageShare("gazebo_ros")

    return LaunchDescription([
        DeclareLaunchArgument("weights", description="path to .pt checkpoint"),
        DeclareLaunchArgument("num_agents", default_value="6"),
        DeclareLaunchArgument("max_agents", default_value=str(MAX_AGENTS)),
        DeclareLaunchArgument("world", default_value=default_world),
        DeclareLaunchArgument("use_teleop", default_value="true"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([pkg_gazebo_ros, "launch", "gzserver.launch.py"])
            ),
            launch_arguments={"world": LaunchConfiguration("world")}.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([pkg_gazebo_ros, "launch", "gzclient.launch.py"])
            ),
        ),

        OpaqueFunction(function=_build_fleet),
    ])

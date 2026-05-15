#!/usr/bin/env python3
"""
markers_node: publishes a static visualization_msgs/MarkerArray containing
the spawn line, goal line, and 4 perimeter walls from circle_arena.world,
so rviz2 can render them.

rviz only renders TF + RobotModel + Marker geometry, never Gazebo world
models. This node bridges the gap by mirroring the world's static
geometry as Markers on /arena_markers.

QoS uses TRANSIENT_LOCAL so a late-starting rviz still receives the
markers once it subscribes. Also republishes every second as belt-and-
braces in case any subscriber prunes them.
"""
from __future__ import annotations

import pathlib
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from visualization_msgs.msg import Marker, MarkerArray

# code/ shim — get arena constants straight from contract.py.
_HERE = pathlib.Path(__file__).resolve()
_REPO = _HERE.parents[3]
sys.path.insert(0, str(_REPO / "code"))
from contract import GOAL_Y, SPAWN_Y, WORLD_H, WORLD_W  # noqa: E402


def _box_marker(idx: int, name: str, pos, size, rgba, frame_id: str = "world") -> Marker:
    m = Marker()
    m.header.frame_id = frame_id
    m.ns = name
    m.id = idx
    m.type = Marker.CUBE
    m.action = Marker.ADD
    m.pose.position.x = float(pos[0])
    m.pose.position.y = float(pos[1])
    m.pose.position.z = float(pos[2])
    m.pose.orientation.w = 1.0
    m.scale.x = float(size[0])
    m.scale.y = float(size[1])
    m.scale.z = float(size[2])
    m.color.r = float(rgba[0])
    m.color.g = float(rgba[1])
    m.color.b = float(rgba[2])
    m.color.a = float(rgba[3])
    m.frame_locked = True
    return m


class MarkersNode(Node):
    def __init__(self) -> None:
        super().__init__("markers_node")

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub = self.create_publisher(MarkerArray, "/arena_markers", qos)

        # Geometry mirrors worlds/circle_arena.world.
        # Lines are 0.01 m thick (vs the world's 0.002 m) so they're visible
        # at default rviz orbit distances; physics doesn't care.
        spawn_line = _box_marker(
            0, "spawn_line",
            pos=(0.0, SPAWN_Y, 0.005),
            size=(WORLD_W, 0.05, 0.01),
            rgba=(0.1, 0.3, 0.95, 1.0),
        )
        goal_line = _box_marker(
            1, "goal_line",
            pos=(0.0, GOAL_Y, 0.005),
            size=(WORLD_W, 0.05, 0.01),
            rgba=(0.15, 0.85, 0.25, 1.0),
        )

        # Walls: 0.1 m thick, 1.0 m tall, centered at ±4.05 m on each axis.
        # Wall material sits just OUTSIDE the trained 8x8 m playable
        # boundary so the floor is exactly 8x8 m. Height 1.0 m because
        # robot-robot collisions can launch one upward; 0.4 m walls let
        # them clear. Must match worlds/circle_arena.world exactly.
        wall_thick = 0.1
        wall_height = 1.0
        wall_z = wall_height / 2.0
        wall_offset = 4.05
        wall_color = (0.55, 0.55, 0.55, 0.85)
        ns_len = 2.0 * wall_offset + wall_thick   # 8.2 m
        ew_len = 2.0 * wall_offset - wall_thick   # 8.0 m
        walls = [
            _box_marker(
                2, "wall_north",
                pos=(0.0, wall_offset, wall_z),
                size=(ns_len, wall_thick, wall_height),
                rgba=wall_color,
            ),
            _box_marker(
                3, "wall_south",
                pos=(0.0, -wall_offset, wall_z),
                size=(ns_len, wall_thick, wall_height),
                rgba=wall_color,
            ),
            _box_marker(
                4, "wall_east",
                pos=(wall_offset, 0.0, wall_z),
                size=(wall_thick, ew_len, wall_height),
                rgba=wall_color,
            ),
            _box_marker(
                5, "wall_west",
                pos=(-wall_offset, 0.0, wall_z),
                size=(wall_thick, ew_len, wall_height),
                rgba=wall_color,
            ),
        ]

        self._arr = MarkerArray()
        self._arr.markers = [spawn_line, goal_line, *walls]

        # Publish once now + every 1 s. TRANSIENT_LOCAL handles the
        # common case; the timer is paranoia against subscribers that
        # somehow drop the message.
        self._publish()
        self.create_timer(1.0, self._publish)

        self.get_logger().info(
            f"markers_node up: {len(self._arr.markers)} markers on /arena_markers "
            f"(WORLD_W={WORLD_W}, SPAWN_Y={SPAWN_Y}, GOAL_Y={GOAL_Y})"
        )

    def _publish(self) -> None:
        stamp = self.get_clock().now().to_msg()
        for m in self._arr.markers:
            m.header.stamp = stamp
        self.pub.publish(self._arr)


def main(args=None):
    rclpy.init(args=args)
    node = MarkersNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

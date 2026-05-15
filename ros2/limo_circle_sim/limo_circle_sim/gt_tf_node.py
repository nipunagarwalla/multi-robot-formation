#!/usr/bin/env python3
"""
gt_tf_node — publish ground-truth TFs for every LIMO from /model_states.

rviz used to render each robot via:
    world  →(static, at spawn pose)→  limo_<i>/odom
           →(from gazebo_ros_diff_drive, computed from wheel rotations)→
                                      limo_<i>/base_footprint

The diff_drive plugin's odometry comes from wheel rotation. When a wheel
grinds against a wall (or any constraint), Gazebo's solver pins the body
but the wheel keeps rotating, so the plugin keeps incrementing the odom
estimate even though the robot isn't moving. The TF tree diverges from
physics and rviz draws the robot somewhere it isn't.

This node side-steps that by listening to /model_states (the actual
Gazebo world pose of every entity, 50 Hz from gazebo_ros_state) and
broadcasting world → limo_<i>/base_footprint directly. To avoid a
two-parent TF conflict, the URDF disables the diff_drive plugin's
publish_odom_tf so the encoder-derived chain stops contributing.

circle_node already reads /model_states for the policy's observations,
so it's unaffected by this change — only rviz gets healthier inputs.
"""
from __future__ import annotations

import re

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


# Match only entities whose name is exactly limo_<digits>. Static models
# (walls, lines, ground_plane) and any other future entities are ignored.
_LIMO_NAME = re.compile(r"^limo_\d+$")


class GroundTruthTFNode(Node):
    def __init__(self) -> None:
        super().__init__("gt_tf_node")
        self.declare_parameter("entity_prefix", "limo_")
        self.entity_prefix = str(self.get_parameter("entity_prefix").value)

        self._br = TransformBroadcaster(self)

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.create_subscription(
            ModelStates, "/model_states", self._on_model_states, qos
        )

        self.get_logger().info(
            "gt_tf_node up: republishing /model_states poses as "
            "world → <ns>/base_footprint TFs."
        )

    def _on_model_states(self, msg: ModelStates) -> None:
        stamp = self.get_clock().now().to_msg()
        for i, name in enumerate(msg.name):
            if not _LIMO_NAME.match(name):
                continue
            pose = msg.pose[i]
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = "world"
            t.child_frame_id = f"{name}/base_footprint"
            t.transform.translation.x = float(pose.position.x)
            t.transform.translation.y = float(pose.position.y)
            t.transform.translation.z = float(pose.position.z)
            t.transform.rotation.x = float(pose.orientation.x)
            t.transform.rotation.y = float(pose.orientation.y)
            t.transform.rotation.z = float(pose.orientation.z)
            t.transform.rotation.w = float(pose.orientation.w)
            self._br.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthTFNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ROS2 port of code/teleop.py:KeyboardTeleop. Owns a pygame window for key
input and publishes the world-frame teleop state (mask + per-robot
velocity overrides + present mask for spawn/delete) for circle_node to consume.

Keys (same as code/teleop.py:KeyboardTeleop):
  1-9        toggle teleop on robot 1-9
  0          toggle teleop on robot 10
  W A S D    drive the most-recently-selected teleop robot (world frame)
  Z / X      decrease / increase teleop drive speed
  =  / +     spawn a new robot (flips a free present_mask slot to 1)
  -  / _     delete selected/last robot (flips present_mask slot to 0)
  R          release all teleop'd robots
  ESC / Q    quit
"""
from __future__ import annotations

import os
import pathlib
import sys

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Float32MultiArray

# code/ shim for contract constants
_HERE = pathlib.Path(__file__).resolve()
_REPO = _HERE.parents[3]
sys.path.insert(0, str(_REPO / "code"))
from contract import MAX_AGENTS, MIN_AGENTS, MAX_V  # noqa: E402


DRIVE_KEYS_MAP = {
    "w": np.array([0.0, 1.0], dtype=np.float32),
    "s": np.array([0.0, -1.0], dtype=np.float32),
    "a": np.array([-1.0, 0.0], dtype=np.float32),
    "d": np.array([1.0, 0.0], dtype=np.float32),
}
SPEED_STEP = 0.25
SPEED_MIN = 0.25
SPEED_MAX = MAX_V


class TeleopNode(Node):
    def __init__(self) -> None:
        super().__init__("teleop_node")
        self.declare_parameter("num_agents", 6)
        self.declare_parameter("drive_speed", float(MAX_V))
        self.declare_parameter("tick_hz", 30.0)

        n0 = int(self.get_parameter("num_agents").value)
        self.drive_speed = float(self.get_parameter("drive_speed").value)
        tick_hz = float(self.get_parameter("tick_hz").value)

        self.n = MAX_AGENTS
        self.present_mask = np.zeros(self.n, dtype=np.float32)
        self.present_mask[:n0] = 1.0
        self.teleop_mask = np.zeros(self.n, dtype=np.float32)
        self.teleop_vels = np.zeros((self.n, 2), dtype=np.float32)
        self.selected: int | None = None
        self.pressed: set[str] = set()

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.pub_cmd = self.create_publisher(Float32MultiArray, "/teleop/cmd", qos)
        self.pub_mask = self.create_publisher(Float32MultiArray, "/teleop/mask", qos)
        self.pub_present = self.create_publisher(Float32MultiArray, "/teleop/present", qos)

        # pygame init last so we fail fast on bad params before opening a window
        import pygame
        os.environ.setdefault("SDL_VIDEO_CENTERED", "1")
        pygame.init()
        pygame.display.set_caption("limo_circle_sim teleop")
        self.screen = pygame.display.set_mode((420, 360))
        self.font = pygame.font.SysFont("monospace", 14)
        self.pygame = pygame
        self.clock = pygame.time.Clock()
        self.tick_hz = tick_hz

        self.timer = self.create_timer(1.0 / tick_hz, self._tick)
        self.get_logger().info(
            f"teleop_node up: n0={n0} drive_speed={self.drive_speed} @ {tick_hz} Hz"
        )

    # ----------------------------------------------------- key actions
    def _key_to_robot(self, ch: str) -> int | None:
        if ch in "123456789":
            return int(ch) - 1
        if ch == "0":
            return 9
        return None

    def _toggle(self, robot: int) -> None:
        if not (0 <= robot < self.n):
            return
        if self.present_mask[robot] < 0.5:
            return
        active = self.teleop_mask[robot] > 0.5
        self.teleop_mask[robot] = 0.0 if active else 1.0
        if not active:
            self.selected = robot
        elif self.selected == robot:
            self.selected = None

    def _release_all(self) -> None:
        self.teleop_mask[:] = 0.0
        self.teleop_vels[:] = 0.0
        self.selected = None

    def _spawn(self) -> None:
        free = np.where(self.present_mask < 0.5)[0]
        if len(free) == 0:
            return
        i = int(free[0])
        self.present_mask[i] = 1.0

    def _delete(self) -> None:
        n_present = int(self.present_mask.sum())
        if n_present <= MIN_AGENTS:
            return
        target = self.selected
        if target is None or self.present_mask[target] < 0.5:
            present = np.where(self.present_mask > 0.5)[0]
            if len(present) == 0:
                return
            target = int(present[-1])
        self.present_mask[target] = 0.0
        self.teleop_mask[target] = 0.0
        self.teleop_vels[target] = 0.0
        if self.selected == target:
            self.selected = None

    # ----------------------------------------------------- main loop
    def _pump_events(self) -> bool:
        """Return False if the user asked to quit."""
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                ch = event.unicode.lower() if event.unicode else ""
                if ch in DRIVE_KEYS_MAP:
                    self.pressed.add(ch)
                    continue
                robot = self._key_to_robot(ch) if ch else None
                if robot is not None:
                    self._toggle(robot)
                elif ch == "r":
                    self._release_all()
                elif ch in ("=", "+"):
                    self._spawn()
                elif ch in ("-", "_"):
                    self._delete()
                elif ch == "z":
                    self.drive_speed = max(SPEED_MIN, self.drive_speed - SPEED_STEP)
                elif ch == "x":
                    self.drive_speed = min(SPEED_MAX, self.drive_speed + SPEED_STEP)
            elif event.type == pygame.KEYUP:
                ch = event.unicode.lower() if event.unicode else ""
                if ch in DRIVE_KEYS_MAP:
                    self.pressed.discard(ch)
        return True

    def _compute_teleop_vels(self) -> None:
        """Mirror code/teleop.py:KeyboardTeleop.apply()."""
        v = np.zeros(2, dtype=np.float32)
        if self.selected is not None and self.pressed:
            for k in self.pressed:
                v += DRIVE_KEYS_MAP[k]
            norm = float(np.linalg.norm(v))
            if norm > 0:
                v = v / norm * self.drive_speed
        for r in range(self.n):
            if self.teleop_mask[r] > 0.5:
                if r == self.selected:
                    self.teleop_vels[r] = v
                else:
                    self.teleop_vels[r] = 0.0
            else:
                self.teleop_vels[r] = 0.0

    def _publish(self) -> None:
        m = Float32MultiArray()
        m.data = self.present_mask.tolist()
        self.pub_present.publish(m)
        m = Float32MultiArray()
        m.data = self.teleop_mask.tolist()
        self.pub_mask.publish(m)
        m = Float32MultiArray()
        m.data = self.teleop_vels.flatten().tolist()
        self.pub_cmd.publish(m)

    def _draw(self) -> None:
        pygame = self.pygame
        self.screen.fill((20, 20, 24))
        lines = [
            f"speed: {self.drive_speed:.2f} m/s    selected: {self.selected}",
            f"n_present: {int(self.present_mask.sum())}    n_teleop: {int(self.teleop_mask.sum())}",
            "",
        ]
        for i in range(self.n):
            tag = "T" if self.teleop_mask[i] > 0.5 else ("P" if self.present_mask[i] > 0.5 else ".")
            sel = " <" if self.selected == i else ""
            lines.append(f"  robot {i+1:2d}  [{tag}]{sel}")
        lines.append("")
        lines.append("keys: 1-9/0 toggle | WASD drive | Z/X speed")
        lines.append("      = spawn | - delete | R release | Esc quit")
        for k, line in enumerate(lines):
            surf = self.font.render(line, True, (220, 220, 220))
            self.screen.blit(surf, (10, 10 + 16 * k))
        pygame.display.flip()

    def _tick(self) -> None:
        if not self._pump_events():
            self.get_logger().info("teleop quit requested")
            rclpy.shutdown()
            return
        self._compute_teleop_vels()
        self._publish()
        self._draw()
        self.clock.tick(int(self.tick_hz))

    def destroy_node(self) -> bool:
        try:
            self.pygame.quit()
        except Exception:
            pass
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

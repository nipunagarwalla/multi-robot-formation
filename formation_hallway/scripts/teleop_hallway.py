#!/usr/bin/python3
"""
Toggle-style multi-robot teleop, matching pygame KeyboardTeleop.

Keys:
  1, 2, 3, 4   -> toggle teleop on triton_1..4 (selects last toggled-on)
  W / A / S / D -> drive the SELECTED robot (world-frame)
  0            -> release ALL teleops, zero everyone
  ESC          -> quit

Publishes:
  /teleop_mask                 std_msgs/Float32MultiArray, length=4
                               (1.0 = teleop'd, 0.0 = policy)
  /triton_<i>/cmd_vel          geometry_msgs/Twist, ONLY for teleop'd robots.
                               The env node skips publishing for any robot
                               with mask=1, so the two never fight.

WASD vectors are world-frame [vx, vy] in m/s. We read each robot's odom
yaw and rotate world->body before publishing to the model_push plugin
(which expects body-frame Twist).

Run after the launch file is up:
  rosrun formation_hallway teleop_hallway.py
"""

from __future__ import annotations

import math
import os
import sys
import threading

import rospy
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Float32MultiArray, MultiArrayDimension

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "config"))
from contract import MAX_AGENTS, MAX_V  # noqa: E402

PUBLISH_HZ = 20

# WASD -> world-frame unit vector
KEY_VECS = {
    "w": (0.0, +1.0),
    "s": (0.0, -1.0),
    "a": (-1.0, 0.0),
    "d": (+1.0, 0.0),
}


def yaw_from_quat(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


class TeleopHallway:
    def __init__(self):
        self.n = MAX_AGENTS
        self.mask = [0.0] * self.n
        self.selected = None        # int or None
        self.pressed = set()
        self.yaws = [0.0] * self.n
        self.lock = threading.Lock()

        self.cmd_pubs = [
            rospy.Publisher(f"/triton_{i+1}/cmd_vel", Twist, queue_size=1)
            for i in range(self.n)
        ]
        self.mask_pub = rospy.Publisher(
            "/teleop_mask", Float32MultiArray, queue_size=1, latch=True
        )

        rospy.Subscriber("/gazebo/model_states", ModelStates,
                         self._model_states_cb, queue_size=1)

        self._publish_mask()

    # ------------------------------------------------------------------ subs
    def _model_states_cb(self, msg: ModelStates):
        for i in range(self.n):
            name = f"triton_{i+1}"
            try:
                idx = msg.name.index(name)
            except ValueError:
                continue
            q = msg.pose[idx].orientation
            self.yaws[i] = yaw_from_quat(q.x, q.y, q.z, q.w)

    # ------------------------------------------------------------------ pub
    def _publish_mask(self):
        m = Float32MultiArray()
        m.layout.dim = [MultiArrayDimension(label="robots", size=self.n, stride=self.n)]
        m.data = list(self.mask)
        self.mask_pub.publish(m)

    def _publish_cmd_for_held(self):
        # build world-frame [vx, vy] for the selected robot from pressed keys
        vx_w, vy_w = 0.0, 0.0
        for k in self.pressed:
            if k in KEY_VECS:
                ax, ay = KEY_VECS[k]
                vx_w += ax
                vy_w += ay
        # normalize so diagonals don't get sqrt(2)x speed, scale by MAX_V
        norm = math.hypot(vx_w, vy_w)
        if norm > 1e-6:
            vx_w = vx_w / norm * MAX_V
            vy_w = vy_w / norm * MAX_V

        for i in range(self.n):
            if self.mask[i] < 0.5:
                continue
            if i != self.selected:
                # held-but-not-driving: hold in place
                self.cmd_pubs[i].publish(Twist())
                continue
            yaw = self.yaws[i]
            # rotate world -> body to feed model_push (which rotates body -> world)
            lin_x =  vx_w * math.cos(yaw) + vy_w * math.sin(yaw)
            lin_y = -vx_w * math.sin(yaw) + vy_w * math.cos(yaw)
            tw = Twist()
            tw.linear.x = lin_x
            tw.linear.y = lin_y
            tw.angular.z = 0.0  # leave yaw drift to env node / passive
            self.cmd_pubs[i].publish(tw)

    # ------------------------------------------------------------------ keys
    def on_press(self, key):
        with self.lock:
            if key in {"1", "2", "3", "4"}:
                idx = int(key) - 1
                if self.mask[idx] > 0.5:
                    # toggle off
                    self.mask[idx] = 0.0
                    self.cmd_pubs[idx].publish(Twist())  # zero on release
                    if self.selected == idx:
                        # pick another teleop'd robot to be selected, if any
                        others = [i for i, m in enumerate(self.mask) if m > 0.5]
                        self.selected = others[-1] if others else None
                else:
                    # toggle on; this becomes selected
                    self.mask[idx] = 1.0
                    self.selected = idx
                self._publish_mask()
                rospy.loginfo(f"[teleop] mask={self.mask} selected={self.selected}")

            elif key == "0":
                self.mask = [0.0] * self.n
                self.selected = None
                for p in self.cmd_pubs:
                    p.publish(Twist())
                self._publish_mask()
                rospy.loginfo("[teleop] released all")

            elif key in KEY_VECS:
                self.pressed.add(key)

    def on_release(self, key):
        with self.lock:
            if key in KEY_VECS:
                self.pressed.discard(key)

    # ------------------------------------------------------------------ loop
    def spin(self):
        rate = rospy.Rate(PUBLISH_HZ)
        while not rospy.is_shutdown():
            with self.lock:
                self._publish_cmd_for_held()
            rate.sleep()


# --- terminal keyboard reader (pynput-free; uses raw stdin in cbreak mode) ---
def _keyboard_thread(t: TeleopHallway):
    """Reads single keystrokes from stdin and routes them to t.on_press.

    Doesn't distinguish key-down vs key-up, so WASD is "tap-to-pulse" rather
    than "hold-to-drive". The publish loop runs at PUBLISH_HZ regardless;
    each tap just adds a key to the pressed set for ~0.15s before clearing.
    """
    import select
    import termios
    import time
    import tty

    HOLD_S = 0.15
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        last_pressed = {}  # key -> wall time when pressed
        while not rospy.is_shutdown():
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            now = time.time()
            if r:
                ch = sys.stdin.read(1)
                if ch == "\x1b":   # ESC
                    rospy.signal_shutdown("user quit")
                    break
                ch = ch.lower()
                t.on_press(ch)
                if ch in KEY_VECS:
                    last_pressed[ch] = now
            # auto-release stale WASD presses
            stale = [k for k, ts in last_pressed.items() if now - ts > HOLD_S]
            for k in stale:
                t.on_release(k)
                last_pressed.pop(k, None)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    rospy.init_node("teleop_hallway")
    t = TeleopHallway()

    print("[teleop] keys: 1-4 toggle teleop · WASD drive selected · 0 release all · ESC quit")
    th = threading.Thread(target=_keyboard_thread, args=(t,), daemon=True)
    th.start()
    t.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

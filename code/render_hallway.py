"""Pygame renderer for FormationHallwayEnv.

Top-down view of a tall vertical hallway. World +y is "forward" so screen
+y is inverted. Drawn elements:
  - hallway walls (left/right) + goal line at top
  - 4 colored circles (one per robot), with TELEOP rings on overridden robots
  - target-formation outline (faded) at the active-cluster centroid
  - thin arrow from each policy-controlled robot to its assigned slot
"""
from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import numpy as np
import pygame
import torch
from scipy.optimize import linear_sum_assignment

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from contract import (
    AGENT_RADIUS,
    DEFAULT_RENDER_PX_PER_M,
    FORMATION_SCALE,
    GOAL_Y,
    WORLD_H,
    WORLD_W,
)
from env_hallway import FormationHallwayEnv, target_formation_positions


WHITE = (245, 245, 245)
BLACK = (10, 10, 10)
WALL = (60, 60, 60)
GOAL = (0, 180, 0)
SLOT = (180, 180, 200)
TELEOP_RING = (220, 70, 70)
HUD = (40, 40, 40)
ROBOT_COLORS = [
    (50, 110, 200),   # blue
    (200, 130, 30),   # orange
    (30, 160, 80),    # green
    (180, 60, 180),   # magenta
]


class HallwayRenderer:
    def __init__(
        self,
        world_w: float = WORLD_W,
        world_h: float = WORLD_H,
        px_per_m: int = DEFAULT_RENDER_PX_PER_M,
        agent_radius: float = AGENT_RADIUS,
    ):
        self.world_w = world_w
        self.world_h = world_h
        self.px = px_per_m
        self.agent_radius = agent_radius
        self.size = (int(world_w * px_per_m), int(world_h * px_per_m))
        self.surface: Optional[pygame.Surface] = None
        self.font: Optional[pygame.font.Font] = None

    def init(self, headless: bool = False):
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        pygame.font.init()
        self.surface = pygame.display.set_mode(self.size)
        pygame.display.set_caption("FormationHallway")
        self.font = pygame.font.Font(None, 18)
        return self.surface

    def w2s(self, p) -> Tuple[int, int]:
        x, y = float(p[0]), float(p[1])
        sx = int((x + self.world_w / 2) * self.px)
        sy = int((self.world_h / 2 - y) * self.px)
        return sx, sy

    def _draw_hallway(self, surf):
        surf.fill(WHITE)
        wall_thickness = max(2, int(0.04 * self.px))
        # left/right walls
        pygame.draw.line(surf, WALL, self.w2s((-self.world_w / 2, -self.world_h / 2)),
                         self.w2s((-self.world_w / 2, self.world_h / 2)), wall_thickness)
        pygame.draw.line(surf, WALL, self.w2s((self.world_w / 2, -self.world_h / 2)),
                         self.w2s((self.world_w / 2, self.world_h / 2)), wall_thickness)
        # goal line
        pygame.draw.line(surf, GOAL, self.w2s((-self.world_w / 2, GOAL_Y)),
                         self.w2s((self.world_w / 2, GOAL_Y)), 2)

    def _draw_formation_overlay(self, surf, ps_active: np.ndarray, k: int, scale: float):
        if k < 2:
            return
        slots = target_formation_positions(k, scale).cpu().numpy()
        centroid = ps_active.mean(axis=0)
        slots_world = slots + centroid
        # Hungarian-assign so the rings line up with the actual robots
        cost = np.linalg.norm(ps_active[:, None, :] - slots_world[None, :, :], axis=-1)
        row, col = linear_sum_assignment(cost)
        slot_radius = max(2, int(self.agent_radius * self.px * 1.05))
        for ri, ci in zip(row, col):
            sx, sy = self.w2s(slots_world[ci])
            pygame.draw.circle(surf, SLOT, (sx, sy), slot_radius, 2)
        # connect each robot to its slot with a faint line
        for ri, ci in zip(row, col):
            pygame.draw.line(surf, SLOT, self.w2s(ps_active[ri]), self.w2s(slots_world[ci]), 1)

    def _draw_robots(self, surf, ps: np.ndarray, teleop_mask: np.ndarray):
        r_px = max(3, int(self.agent_radius * self.px))
        for i, p in enumerate(ps):
            color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            sx, sy = self.w2s(p)
            pygame.draw.circle(surf, color, (sx, sy), r_px)
            if float(teleop_mask[i]) > 0.5:
                pygame.draw.circle(surf, TELEOP_RING, (sx, sy), r_px + 4, 2)
                if self.font:
                    label = self.font.render("T", True, TELEOP_RING)
                    surf.blit(label, (sx - 4, sy - r_px - 14))
            if self.font:
                idx_label = self.font.render(str(i + 1), True, BLACK)
                surf.blit(idx_label, (sx + r_px + 2, sy - 8))

    def _draw_hud(self, surf, active_count: int, episode_step: int, total_reward: float):
        if not self.font:
            return
        lines = [
            f"active={active_count}  step={episode_step}  R={total_reward:+.2f}",
            "1/2/3/4 toggle teleop · WASD drive · 0 release",
        ]
        for i, line in enumerate(lines):
            surf.blit(self.font.render(line, True, HUD), (8, 8 + 16 * i))

    def render(
        self,
        env: FormationHallwayEnv,
        env_idx: int = 0,
        episode_step: int = 0,
        total_reward: float = 0.0,
        formation_scale: float = FORMATION_SCALE,
    ):
        if self.surface is None:
            self.init()
        self._draw_hallway(self.surface)
        ps = env.ps[env_idx].detach().cpu().numpy()
        teleop = env.teleop_mask[env_idx].detach().cpu().numpy()
        active_idx = np.where(teleop < 0.5)[0]
        ps_active = ps[active_idx]
        self._draw_formation_overlay(self.surface, ps_active, k=len(active_idx),
                                     scale=formation_scale)
        self._draw_robots(self.surface, ps, teleop)
        self._draw_hud(self.surface, active_count=len(active_idx),
                       episode_step=episode_step, total_reward=total_reward)
        pygame.display.flip()

    def close(self):
        if self.surface is not None:
            pygame.display.quit()
            pygame.quit()
            self.surface = None


def _self_test():
    """Cycle through n in {4,3,2,1} with hardcoded poses to verify overlays."""
    env = FormationHallwayEnv({"num_envs": 1})
    env.vector_reset()
    r = HallwayRenderer()
    r.init(headless=True)
    rng = np.random.default_rng(0)
    out_path = os.path.join(os.path.dirname(__file__), "..", "render_self_test.png")
    out_path = os.path.abspath(out_path)
    frames = []
    for n_active in (4, 3, 2, 1):
        # set positions roughly in a vertical column then teleop the trailing ones
        env.vector_reset()
        ps = torch.zeros(env.cfg["n_agents"], 2)
        for i in range(env.cfg["n_agents"]):
            ps[i] = torch.tensor([(-0.3 + 0.2 * i), 0.0])
        env.ps[0] = ps
        for i in range(n_active, env.cfg["n_agents"]):
            env.set_teleop(0, i, True)
            env.set_teleop_action(0, i, np.array([0.0, 0.0]))
        r.render(env, env_idx=0, episode_step=0, total_reward=0.0)
        # capture frame
        arr = pygame.surfarray.array3d(r.surface).swapaxes(0, 1)
        frames.append(arr)
    # stitch the four frames horizontally and save
    h = max(f.shape[0] for f in frames)
    w = sum(f.shape[1] for f in frames)
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    x = 0
    for f in frames:
        canvas[: f.shape[0], x : x + f.shape[1]] = f
        x += f.shape[1]
    surf = pygame.surfarray.make_surface(canvas.swapaxes(0, 1))
    pygame.image.save(surf, out_path)
    print(f"self-test wrote {out_path}  ({len(frames)} frames stitched)")
    r.close()


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    _self_test()

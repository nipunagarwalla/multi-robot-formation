"""Pygame renderer for FormationHallwayEnv.

Top-down view of a tall vertical hallway. World +y is "forward" so screen
+y is inverted. Drawn elements:
  - hallway walls (left/right) drawn as thick filled rectangles
  - distance ticks every 1 m along the inside of the walls
  - spawn line (blue) at SPAWN_Y, goal line (green) at GOAL_Y
  - 4 colored circles (one per robot), with TELEOP rings on overridden robots
  - target-formation outline (faded) at the active-cluster centroid
  - thin line from each policy-controlled robot to its assigned slot
  - HUD in the left margin panel (outside the hallway)

Defaults to fullscreen so the hallway fills the screen vertically. Opt out
with HallwayRenderer(fullscreen=False) for headless self-tests.
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
    SPAWN_Y,
    WORLD_H,
    WORLD_W,
)
from env_hallway import FormationHallwayEnv, target_formation_positions


BG_OUTSIDE = (24, 24, 28)        # dark frame around the hallway
BG_INSIDE = (245, 245, 245)      # hallway floor
WALL = (40, 40, 40)              # filled wall rectangles
WALL_HIGHLIGHT = (180, 180, 190) # thin inner edge
TICK = (140, 140, 150)
TICK_LABEL = (90, 90, 100)
GOAL = (0, 170, 0)
GOAL_LABEL = (0, 120, 0)
SPAWN = (60, 110, 200)
SPAWN_LABEL = (40, 90, 170)
SLOT = (180, 180, 200)
TELEOP_RING = (220, 70, 70)
HUD = (235, 235, 240)
HUD_DIM = (160, 160, 170)
BLACK = (10, 10, 10)
ROBOT_COLORS = [
    (50, 110, 200),   # blue
    (210, 130, 30),   # orange
    (30, 170, 80),    # green
    (190, 60, 190),   # magenta
]


class HallwayRenderer:
    """Fullscreen by default; pass fullscreen=False for windowed/headless."""

    def __init__(
        self,
        world_w: float = WORLD_W,
        world_h: float = WORLD_H,
        px_per_m: int = DEFAULT_RENDER_PX_PER_M,
        agent_radius: float = AGENT_RADIUS,
        fullscreen: bool = True,
        windowed_size: Optional[Tuple[int, int]] = None,
    ):
        self.world_w = world_w
        self.world_h = world_h
        self.agent_radius = agent_radius
        self.fullscreen = fullscreen
        # only used when fullscreen=False; pygame picks display size in fullscreen mode
        self._fallback_px = px_per_m
        self._windowed_size = windowed_size
        # filled by init()
        self.px = px_per_m
        self.screen_w = world_w * px_per_m
        self.screen_h = world_h * px_per_m
        self.hall_left_px = 0     # x-pixel of left wall inner edge
        self.hall_right_px = 0    # x-pixel of right wall inner edge
        self.hall_top_px = 0      # y-pixel of hallway top edge
        self.hall_bottom_px = 0   # y-pixel of hallway bottom edge
        self.surface: Optional[pygame.Surface] = None
        self.font_hud: Optional[pygame.font.Font] = None
        self.font_tick: Optional[pygame.font.Font] = None
        self.font_label: Optional[pygame.font.Font] = None

    # ---- init / geometry -------------------------------------------------
    def init(self, headless: bool = False):
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            self.fullscreen = False
        pygame.init()
        pygame.font.init()
        if self.fullscreen and not headless:
            self.surface = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            self.screen_w, self.screen_h = self.surface.get_size()
        else:
            if self._windowed_size is not None:
                size = self._windowed_size
            else:
                size = (int(self.world_w * self._fallback_px),
                        int(self.world_h * self._fallback_px))
            self.surface = pygame.display.set_mode(size)
            self.screen_w, self.screen_h = size

        # pick px_per_m so the hallway fills 92% of vertical space
        self.px = max(8, int(self.screen_h * 0.92 / self.world_h))
        hall_w_px = int(self.world_w * self.px)
        hall_h_px = int(self.world_h * self.px)
        # center horizontally, near top vertically (with HUD margin both sides)
        self.hall_left_px = (self.screen_w - hall_w_px) // 2
        self.hall_right_px = self.hall_left_px + hall_w_px
        self.hall_top_px = (self.screen_h - hall_h_px) // 2
        self.hall_bottom_px = self.hall_top_px + hall_h_px

        self.font_hud = pygame.font.Font(None, 28)
        self.font_tick = pygame.font.Font(None, 18)
        self.font_label = pygame.font.Font(None, 22)
        pygame.display.set_caption("FormationHallway")
        return self.surface

    def w2s(self, p) -> Tuple[int, int]:
        """world (x, y) -> screen pixel. World +y is up; screen +y is down."""
        x, y = float(p[0]), float(p[1])
        sx = self.hall_left_px + int((x + self.world_w / 2) * self.px)
        sy = self.hall_top_px + int((self.world_h / 2 - y) * self.px)
        return sx, sy

    # ---- drawing pieces --------------------------------------------------
    def _draw_frame(self, surf):
        surf.fill(BG_OUTSIDE)
        # hallway floor
        floor = pygame.Rect(self.hall_left_px, self.hall_top_px,
                            self.hall_right_px - self.hall_left_px,
                            self.hall_bottom_px - self.hall_top_px)
        pygame.draw.rect(surf, BG_INSIDE, floor)

        wall_thick = max(8, int(self.px * 0.12))
        # left wall (sits OUTSIDE the floor so the inner edge is exact)
        left_wall = pygame.Rect(self.hall_left_px - wall_thick, self.hall_top_px,
                                wall_thick, self.hall_bottom_px - self.hall_top_px)
        right_wall = pygame.Rect(self.hall_right_px, self.hall_top_px,
                                 wall_thick, self.hall_bottom_px - self.hall_top_px)
        pygame.draw.rect(surf, WALL, left_wall)
        pygame.draw.rect(surf, WALL, right_wall)
        # 1-px highlight on the inner edge of each wall to emphasize the boundary
        pygame.draw.line(surf, WALL_HIGHLIGHT,
                         (self.hall_left_px, self.hall_top_px),
                         (self.hall_left_px, self.hall_bottom_px), 1)
        pygame.draw.line(surf, WALL_HIGHLIGHT,
                         (self.hall_right_px - 1, self.hall_top_px),
                         (self.hall_right_px - 1, self.hall_bottom_px), 1)

    def _draw_ticks(self, surf):
        if self.font_tick is None:
            return
        tick_len = max(8, int(self.px * 0.10))
        # ticks every integer y in [-world_h/2, +world_h/2]
        y_min = -int(self.world_h // 2)
        y_max = int(self.world_h // 2)
        for y in range(y_min, y_max + 1):
            sx_l, sy = self.w2s((-self.world_w / 2, float(y)))
            sx_r, _ = self.w2s((self.world_w / 2, float(y)))
            pygame.draw.line(surf, TICK, (sx_l, sy), (sx_l + tick_len, sy), 1)
            pygame.draw.line(surf, TICK, (sx_r - tick_len, sy), (sx_r, sy), 1)
            label = self.font_tick.render(f"{y:+d} m", True, TICK_LABEL)
            # left tick label sits just outside the wall in the dark frame
            surf.blit(label, (self.hall_left_px - label.get_width() - 14, sy - 8))

    def _draw_lines_of_interest(self, surf):
        # spawn line (blue)
        sx_l, sy = self.w2s((-self.world_w / 2, SPAWN_Y))
        sx_r, _ = self.w2s((self.world_w / 2, SPAWN_Y))
        pygame.draw.line(surf, SPAWN, (sx_l, sy), (sx_r, sy), 4)
        if self.font_label:
            lbl = self.font_label.render("SPAWN", True, SPAWN_LABEL)
            surf.blit(lbl, (sx_r + 12, sy - 10))

        # goal line (green)
        sx_l, sy = self.w2s((-self.world_w / 2, GOAL_Y))
        sx_r, _ = self.w2s((self.world_w / 2, GOAL_Y))
        pygame.draw.line(surf, GOAL, (sx_l, sy), (sx_r, sy), 4)
        if self.font_label:
            lbl = self.font_label.render("GOAL", True, GOAL_LABEL)
            surf.blit(lbl, (sx_r + 12, sy - 10))

    def _draw_formation_overlay(self, surf, ps_active: np.ndarray, k: int, scale: float):
        if k < 2:
            return
        slots = target_formation_positions(k, scale).cpu().numpy()
        centroid = ps_active.mean(axis=0)
        slots_world = slots + centroid
        cost = np.linalg.norm(ps_active[:, None, :] - slots_world[None, :, :], axis=-1)
        row, col = linear_sum_assignment(cost)
        slot_radius = max(2, int(self.agent_radius * self.px * 1.05))
        for ri, ci in zip(row, col):
            sx, sy = self.w2s(slots_world[ci])
            pygame.draw.circle(surf, SLOT, (sx, sy), slot_radius, 2)
        for ri, ci in zip(row, col):
            pygame.draw.line(surf, SLOT, self.w2s(ps_active[ri]), self.w2s(slots_world[ci]), 1)

    def _draw_robots(self, surf, ps: np.ndarray, teleop_mask: np.ndarray):
        r_px = max(4, int(self.agent_radius * self.px))
        for i, p in enumerate(ps):
            color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            sx, sy = self.w2s(p)
            pygame.draw.circle(surf, color, (sx, sy), r_px)
            if float(teleop_mask[i]) > 0.5:
                pygame.draw.circle(surf, TELEOP_RING, (sx, sy), r_px + 4, 2)
                if self.font_label:
                    lbl = self.font_label.render("T", True, TELEOP_RING)
                    surf.blit(lbl, (sx - 4, sy - r_px - 18))
            if self.font_label:
                idx = self.font_label.render(str(i + 1), True, BLACK)
                surf.blit(idx, (sx + r_px + 4, sy - 10))

    def _draw_hud(
        self,
        surf,
        active_count: int,
        episode_step: int,
        total_reward: float,
        teleop_speed: Optional[float] = None,
    ):
        if self.font_hud is None:
            return
        # left margin panel — anchored to the dark frame, never on the floor
        panel_x = max(12, self.hall_left_px - 240)
        speed_line = (
            f"teleop speed = {teleop_speed:.2f} m/s"
            if teleop_speed is not None
            else ""
        )
        lines = [
            ("FormationHallway", HUD),
            (f"active = {active_count}", HUD),
            (f"step   = {episode_step}", HUD_DIM),
            (f"reward = {total_reward:+.2f}", HUD_DIM),
            (speed_line, HUD_DIM),
            ("", HUD_DIM),
            ("1/2/3/4 toggle teleop", HUD_DIM),
            ("WASD drive  Z/X speed", HUD_DIM),
            ("0 release  ESC quit", HUD_DIM),
        ]
        y = self.hall_top_px
        for text, color in lines:
            if text:
                surf.blit(self.font_hud.render(text, True, color), (panel_x, y))
            y += 30

    # ---- public API ------------------------------------------------------
    def render(
        self,
        env: FormationHallwayEnv,
        env_idx: int = 0,
        episode_step: int = 0,
        total_reward: float = 0.0,
        formation_scale: float = FORMATION_SCALE,
        teleop_speed: Optional[float] = None,
    ):
        if self.surface is None:
            self.init()
        self._draw_frame(self.surface)
        self._draw_ticks(self.surface)
        self._draw_lines_of_interest(self.surface)
        ps = env.ps[env_idx].detach().cpu().numpy()
        teleop = env.teleop_mask[env_idx].detach().cpu().numpy()
        active_idx = np.where(teleop < 0.5)[0]
        ps_active = ps[active_idx]
        self._draw_formation_overlay(self.surface, ps_active, k=len(active_idx),
                                     scale=formation_scale)
        self._draw_robots(self.surface, ps, teleop)
        self._draw_hud(self.surface, active_count=len(active_idx),
                       episode_step=episode_step, total_reward=total_reward,
                       teleop_speed=teleop_speed)
        pygame.display.flip()

    def close(self):
        if self.surface is not None:
            pygame.display.quit()
            pygame.quit()
            self.surface = None


def _self_test():
    """Render n in {4,3,2,1} with hardcoded poses to a stitched png."""
    env = FormationHallwayEnv({"num_envs": 1})
    env.vector_reset()
    r = HallwayRenderer(fullscreen=False, windowed_size=(800, 1000))
    r.init(headless=True)
    out_path = os.path.join(os.path.dirname(__file__), "..", "render_self_test.png")
    out_path = os.path.abspath(out_path)
    frames = []
    for n_active in (4, 3, 2, 1):
        env.vector_reset()
        ps = torch.zeros(env.cfg["n_agents"], 2)
        for i in range(env.cfg["n_agents"]):
            ps[i] = torch.tensor([(-0.3 + 0.2 * i), 0.0])
        env.ps[0] = ps
        for i in range(n_active, env.cfg["n_agents"]):
            env.set_teleop(0, i, True)
            env.set_teleop_action(0, i, np.array([0.0, 0.0]))
        r.render(env, env_idx=0, episode_step=0, total_reward=0.0)
        arr = pygame.surfarray.array3d(r.surface).swapaxes(0, 1)
        frames.append(arr)
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

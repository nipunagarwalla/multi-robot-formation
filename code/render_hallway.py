"""Pygame renderer for FormationHallwayEnv (8 m x 8 m square arena).

Top-down view. World +y is "forward" so screen +y is inverted. Drawn
elements:
  - arena floor + four wall rectangles
  - distance ticks every 1 m along the bottom and left walls
  - spawn line (blue) at SPAWN_Y, goal line (green) at GOAL_Y
  - up to 10 colored circles (one per present robot), with red TELEOP rings
    on overridden robots and a "T" label above
  - target-formation outline (faded circle of n_active slots) at the active
    centroid, with thin assignment lines from each policy-controlled robot
  - HUD in the left margin showing n_present / n_active / n_teleop, the
    current step + reward, and key bindings.

Defaults to fullscreen so the arena fills the window. Opt out with
HallwayRenderer(fullscreen=False) for headless self-tests.
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
    CIRCLE_SIDE,
    DEFAULT_RENDER_PX_PER_M,
    GOAL_Y,
    SPAWN_Y,
    WORLD_H,
    WORLD_W,
)
from env_hallway import FormationHallwayEnv, target_formation_positions


BG_OUTSIDE = (24, 24, 28)
BG_INSIDE = (245, 245, 245)
WALL = (40, 40, 40)
WALL_HIGHLIGHT = (180, 180, 190)
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
    (50, 110, 200),   # 1 blue
    (210, 130, 30),   # 2 orange
    (30, 170, 80),    # 3 green
    (190, 60, 190),   # 4 magenta
    (200, 50, 50),    # 5 red
    (60, 180, 200),   # 6 cyan
    (200, 180, 30),   # 7 yellow
    (130, 80, 50),    # 8 brown
    (140, 100, 200),  # 9 purple
    (90, 90, 90),     # 10 grey
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
        self._fallback_px = px_per_m
        self._windowed_size = windowed_size
        self.px = px_per_m
        self.screen_w = world_w * px_per_m
        self.screen_h = world_h * px_per_m
        self.arena_left_px = 0
        self.arena_right_px = 0
        self.arena_top_px = 0
        self.arena_bottom_px = 0
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

        # Pick px_per_m so the arena fits in 92% of the smaller axis (square arena).
        smaller = min(self.screen_w, self.screen_h)
        self.px = max(8, int(smaller * 0.92 / max(self.world_w, self.world_h)))
        arena_w_px = int(self.world_w * self.px)
        arena_h_px = int(self.world_h * self.px)
        # center both ways
        self.arena_left_px = (self.screen_w - arena_w_px) // 2
        self.arena_right_px = self.arena_left_px + arena_w_px
        self.arena_top_px = (self.screen_h - arena_h_px) // 2
        self.arena_bottom_px = self.arena_top_px + arena_h_px

        self.font_hud = pygame.font.Font(None, 28)
        self.font_tick = pygame.font.Font(None, 18)
        self.font_label = pygame.font.Font(None, 22)
        pygame.display.set_caption("FormationHallway")
        return self.surface

    def w2s(self, p) -> Tuple[int, int]:
        """world (x, y) -> screen pixel. World +y is up; screen +y is down."""
        x, y = float(p[0]), float(p[1])
        sx = self.arena_left_px + int((x + self.world_w / 2) * self.px)
        sy = self.arena_top_px + int((self.world_h / 2 - y) * self.px)
        return sx, sy

    # ---- drawing pieces --------------------------------------------------
    def _draw_frame(self, surf):
        surf.fill(BG_OUTSIDE)
        floor = pygame.Rect(self.arena_left_px, self.arena_top_px,
                            self.arena_right_px - self.arena_left_px,
                            self.arena_bottom_px - self.arena_top_px)
        pygame.draw.rect(surf, BG_INSIDE, floor)

        wall_thick = max(8, int(self.px * 0.05))
        # four walls — sit OUTSIDE the floor so the inner edge is exact
        left = pygame.Rect(self.arena_left_px - wall_thick, self.arena_top_px - wall_thick,
                           wall_thick, self.arena_bottom_px - self.arena_top_px + 2 * wall_thick)
        right = pygame.Rect(self.arena_right_px, self.arena_top_px - wall_thick,
                            wall_thick, self.arena_bottom_px - self.arena_top_px + 2 * wall_thick)
        top = pygame.Rect(self.arena_left_px - wall_thick, self.arena_top_px - wall_thick,
                          self.arena_right_px - self.arena_left_px + 2 * wall_thick, wall_thick)
        bottom = pygame.Rect(self.arena_left_px - wall_thick, self.arena_bottom_px,
                             self.arena_right_px - self.arena_left_px + 2 * wall_thick, wall_thick)
        for w in (left, right, top, bottom):
            pygame.draw.rect(surf, WALL, w)
        # 1-px highlight on the inner edge
        pygame.draw.rect(surf, WALL_HIGHLIGHT, floor, 1)

    def _draw_ticks(self, surf):
        if self.font_tick is None:
            return
        tick_len = max(8, int(self.px * 0.10))
        y_min = -int(self.world_h // 2)
        y_max = int(self.world_h // 2)
        for y in range(y_min, y_max + 1):
            sx_l, sy = self.w2s((-self.world_w / 2, float(y)))
            sx_r, _ = self.w2s((self.world_w / 2, float(y)))
            pygame.draw.line(surf, TICK, (sx_l, sy), (sx_l + tick_len, sy), 1)
            pygame.draw.line(surf, TICK, (sx_r - tick_len, sy), (sx_r, sy), 1)
            label = self.font_tick.render(f"{y:+d} m", True, TICK_LABEL)
            surf.blit(label, (self.arena_left_px - label.get_width() - 14, sy - 8))
        x_min = -int(self.world_w // 2)
        x_max = int(self.world_w // 2)
        for x in range(x_min, x_max + 1):
            sx, sy_t = self.w2s((float(x), self.world_h / 2))
            _, sy_b = self.w2s((float(x), -self.world_h / 2))
            pygame.draw.line(surf, TICK, (sx, sy_b - tick_len), (sx, sy_b), 1)

    def _draw_lines_of_interest(self, surf):
        sx_l, sy = self.w2s((-self.world_w / 2, SPAWN_Y))
        sx_r, _ = self.w2s((self.world_w / 2, SPAWN_Y))
        pygame.draw.line(surf, SPAWN, (sx_l, sy), (sx_r, sy), 4)
        if self.font_label:
            lbl = self.font_label.render("SPAWN", True, SPAWN_LABEL)
            surf.blit(lbl, (sx_r + 12, sy - 10))

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
        # faint full-circle outline so the user sees the target shape even
        # before assignments are read
        cx, cy = self.w2s(centroid)
        if k >= 3:
            r_world = scale / (2.0 * np.sin(np.pi / k))
            pygame.draw.circle(surf, SLOT, (cx, cy), int(r_world * self.px), 1)
        for ri, ci in zip(row, col):
            sx, sy = self.w2s(slots_world[ci])
            pygame.draw.circle(surf, SLOT, (sx, sy), slot_radius, 2)
        for ri, ci in zip(row, col):
            pygame.draw.line(surf, SLOT, self.w2s(ps_active[ri]), self.w2s(slots_world[ci]), 1)

    def _draw_robots(self, surf, ps: np.ndarray, teleop_mask: np.ndarray, present_mask: np.ndarray):
        r_px = max(4, int(self.agent_radius * self.px))
        for i, p in enumerate(ps):
            if float(present_mask[i]) < 0.5:
                continue  # skip non-present (parked at sentinel anyway)
            color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
            sx, sy = self.w2s(p)
            pygame.draw.circle(surf, color, (sx, sy), r_px)
            if float(teleop_mask[i]) > 0.5:
                pygame.draw.circle(surf, TELEOP_RING, (sx, sy), r_px + 4, 2)
                if self.font_label:
                    lbl = self.font_label.render("T", True, TELEOP_RING)
                    surf.blit(lbl, (sx - 4, sy - r_px - 18))
            if self.font_label:
                idx = self.font_label.render(str(i + 1) if i < 9 else "0", True, BLACK)
                surf.blit(idx, (sx + r_px + 4, sy - 10))

    def _draw_hud(self, surf, *, n_present: int, n_active: int, n_teleop: int,
                  episode_step: int, total_reward: float, drive_speed: float):
        if self.font_hud is None:
            return
        panel_x = max(12, self.arena_left_px - 240)
        lines = [
            ("FormationHallway", HUD),
            (f"present = {n_present}", HUD),
            (f"active  = {n_active}", HUD),
            (f"teleop  = {n_teleop}", HUD),
            (f"step    = {episode_step}", HUD_DIM),
            (f"reward  = {total_reward:+.2f}", HUD_DIM),
            (f"drive   = {drive_speed:.2f} m/s", HUD_DIM),
            ("", HUD_DIM),
            ("1-9 / 0 toggle teleop", HUD_DIM),
            ("WASD drive  ·  Z / X speed", HUD_DIM),
            ("= spawn   -  delete   R release", HUD_DIM),
            ("ESC quit", HUD_DIM),
        ]
        y = self.arena_top_px
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
        circle_side: float = CIRCLE_SIDE,
        drive_speed: float = 1.0,
    ):
        if self.surface is None:
            self.init()
        self._draw_frame(self.surface)
        self._draw_ticks(self.surface)
        self._draw_lines_of_interest(self.surface)
        ps = env.ps[env_idx].detach().cpu().numpy()
        teleop = env.teleop_mask[env_idx].detach().cpu().numpy()
        present = env.present_mask[env_idx].detach().cpu().numpy()
        active_idx = np.where((present > 0.5) & (teleop < 0.5))[0]
        ps_active = ps[active_idx]
        self._draw_formation_overlay(self.surface, ps_active, k=len(active_idx),
                                     scale=circle_side)
        self._draw_robots(self.surface, ps, teleop, present)
        self._draw_hud(
            self.surface,
            n_present=int(present.sum()),
            n_active=len(active_idx),
            n_teleop=int(teleop.sum()),
            episode_step=episode_step,
            total_reward=total_reward,
            drive_speed=drive_speed,
        )
        pygame.display.flip()

    def close(self):
        if self.surface is not None:
            pygame.display.quit()
            pygame.quit()
            self.surface = None


def _self_test():
    """Render n_present in a few values to a stitched png for visual check."""
    env = FormationHallwayEnv({"num_envs": 1, "initial_agents": 4})
    env.vector_reset()
    r = HallwayRenderer(fullscreen=False, windowed_size=(900, 900))
    r.init(headless=True)
    out_path = os.path.join(os.path.dirname(__file__), "..", "render_self_test.png")
    out_path = os.path.abspath(out_path)
    frames = []
    for n_target in (4, 7, 10, 2, 1):
        env.vector_reset()
        cur = env.n_present(0)
        while cur < n_target:
            env.spawn(0)
            cur += 1
        while cur > n_target:
            # delete from highest index
            for i in range(env.cfg["n_agents"] - 1, -1, -1):
                if float(env.present_mask[0, i]) > 0.5:
                    if env.delete(0, i):
                        cur -= 1
                    break
        # place all present robots in target circle for a clean visual
        slots = target_formation_positions(n_target)
        ps = env.ps[0].clone()
        present_idx = [i for i in range(env.cfg["n_agents"]) if float(env.present_mask[0, i]) > 0.5]
        for slot_i, robot_i in enumerate(present_idx):
            ps[robot_i] = slots[slot_i]
        env.ps[0] = ps
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

"""
arena.py — 2D grid arena with walls, static/dynamic obstacles, ArUco ground truth.
Grid units = cells. Each cell = CELL_SIZE cm real-world equivalent.
"""

import numpy as np
import random

ARENA_W = 40       # cells wide
ARENA_H = 40       # cells tall
CELL_SIZE = 0.5    # metres per cell

# Occupancy values
FREE     = 0
OCCUPIED = 1
UNKNOWN  = -1


class Obstacle:
    def __init__(self, x, y, w=2, h=2, dynamic=False):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.dynamic = dynamic
        self.move_timer = 0
        self.move_interval = random.randint(40, 80)  # steps between moves
        # direction for dynamic obstacles
        self.dx = random.choice([-1, 0, 1])
        self.dy = random.choice([-1, 0, 1])
        while self.dx == 0 and self.dy == 0:
            self.dx = random.choice([-1, 0, 1])
            self.dy = random.choice([-1, 0, 1])

    def cells(self):
        """Return list of (x,y) cells this obstacle occupies."""
        return [
            (self.x + dx, self.y + dy)
            for dx in range(self.w)
            for dy in range(self.h)
        ]


class Arena:
    """
    Manages ground-truth occupancy map, obstacles, ArUco position noise.
    """

    def __init__(self, width=ARENA_W, height=ARENA_H,
                 n_static=4, n_dynamic=3,
                 aruco_noise_sigma=0.01,
                 seed=None):
        self.width = width
        self.height = height
        self.aruco_sigma = aruco_noise_sigma  # metres
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        # Ground-truth binary map (walls=1, free=0)
        self.ground_truth = np.zeros((height, width), dtype=np.int8)

        # Border walls
        self.ground_truth[0, :]  = OCCUPIED
        self.ground_truth[-1, :] = OCCUPIED
        self.ground_truth[:, 0]  = OCCUPIED
        self.ground_truth[:, -1] = OCCUPIED

        # Interior wall segments (hardcoded L-shapes for realism)
        self._place_interior_walls()

        # Static + dynamic obstacles (avoid robot spawn zones top-left 5x5)
        self.obstacles = []
        self._place_obstacles(n_static, n_dynamic)

        self.step_count = 0

    # ------------------------------------------------------------------
    def _place_interior_walls(self):
        """Add a few interior wall segments to make mapping non-trivial."""
        segments = [
            # (col_start, row, length, horizontal)
            (10, 15, 8, True),
            (25, 8,  6, True),
            (5,  25, 6, False),
            (30, 20, 8, False),
        ]
        for c, r, length, horiz in segments:
            for i in range(length):
                col = c + (i if horiz else 0)
                row = r + (0 if horiz else i)
                if 1 <= row < self.height - 1 and 1 <= col < self.width - 1:
                    self.ground_truth[row, col] = OCCUPIED

    def _place_obstacles(self, n_static, n_dynamic):
        spawn_exclusion = set(
            (x, y) for x in range(8) for y in range(8)
        )
        for _ in range(n_static + n_dynamic):
            placed = False
            for attempt in range(200):
                x = random.randint(2, self.width  - 5)
                y = random.randint(2, self.height - 5)
                is_dyn = len(self.obstacles) >= n_static
                obs = Obstacle(x, y, w=2, h=2, dynamic=is_dyn)
                cells = obs.cells()
                if any(c in spawn_exclusion for c in cells):
                    continue
                if any(self.ground_truth[cy, cx] == OCCUPIED
                       for cx, cy in cells
                       if 0 <= cx < self.width and 0 <= cy < self.height):
                    continue
                self.obstacles.append(obs)
                placed = True
                break
            # silently skip if no valid position found

    # ------------------------------------------------------------------
    def _apply_obstacles_to_map(self, grid):
        for obs in self.obstacles:
            for cx, cy in obs.cells():
                if 0 <= cx < self.width and 0 <= cy < self.height:
                    grid[cy, cx] = OCCUPIED

    def get_full_map(self):
        """Ground truth map including current obstacle positions."""
        m = self.ground_truth.copy()
        self._apply_obstacles_to_map(m)
        return m

    def is_occupied(self, cx, cy):
        if cx < 0 or cy < 0 or cx >= self.width or cy >= self.height:
            return True
        m = self.get_full_map()
        return m[cy, cx] == OCCUPIED

    # ------------------------------------------------------------------
    def update(self):
        """Step dynamic obstacles."""
        self.step_count += 1
        full = self.ground_truth.copy()  # walls only for collision check

        for obs in self.obstacles:
            if not obs.dynamic:
                continue
            obs.move_timer += 1
            if obs.move_timer < obs.move_interval:
                continue
            obs.move_timer = 0

            nx = obs.x + obs.dx
            ny = obs.y + obs.dy
            # Check boundary and wall collision
            valid = True
            for dx in range(obs.w):
                for dy in range(obs.h):
                    cx, cy = nx + dx, ny + dy
                    if cx < 1 or cy < 1 or cx >= self.width - 1 or cy >= self.height - 1:
                        valid = False
                    elif full[cy, cx] == OCCUPIED:
                        valid = False
            if valid:
                obs.x = nx
                obs.y = ny
            else:
                # Bounce direction
                obs.dx = -obs.dx
                obs.dy = -obs.dy

    # ------------------------------------------------------------------
    def aruco_observe(self, true_x_m, true_y_m):
        """
        Simulate overhead ArUco observation.
        true_x_m, true_y_m in metres.
        Returns noisy (x, y) in metres.
        """
        nx = true_x_m + self.rng.normal(0, self.aruco_sigma)
        ny = true_y_m + self.rng.normal(0, self.aruco_sigma)
        return float(nx), float(ny)

    # ------------------------------------------------------------------
    def cells_to_metres(self, cx, cy):
        return cx * CELL_SIZE, cy * CELL_SIZE

    def metres_to_cells(self, x_m, y_m):
        return int(x_m / CELL_SIZE), int(y_m / CELL_SIZE)
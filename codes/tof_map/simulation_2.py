"""
robot.py — Robot class.

Sensors (simulated):
  - Wheel odometry (Gaussian noise)
  - IMU heading (Gaussian noise)
  - ToF sweep: 8 ray directions around robot, range-limited
  - ArUco correction every ARUCO_INTERVAL steps

Navigation: Bug2 algorithm
  - Head straight to goal
  - If blocked → wall-follow (right-hand rule)
  - Resume straight line when m-line clear ahead

Mapping: log-odds occupancy grid per robot
  P(occ|hit) = 0.7  →  L_occ =  0.847
  P(occ|free) = 0.3 →  L_free = -0.357

State machine: IDLE → ASSIGNED → NAVIGATING → SCANNING → ZONE_COMPLETE → IDLE
"""

import numpy as np
import math
from arena import ARENA_W, ARENA_H, CELL_SIZE, FREE, OCCUPIED, UNKNOWN

# Log-odds constants
L_OCC   =  math.log(0.7 / 0.3)   #  0.847
L_FREE  =  math.log(0.3 / 0.7)   # -0.847
L_PRIOR =  0.0
L_MIN   = -5.0
L_MAX   =  5.0

# Sensor params
TOF_RANGE_CELLS    = 8      # max ToF range in cells
TOF_ANGLES         = 8      # number of ray directions (every 45°)
ARUCO_INTERVAL     = 5      # correct pose every N steps
SCAN_TURNS         = 1      # full 360° sweeps per SCANNING state

# Navigation params
MOVE_SPEED         = 1      # cells per step
GOAL_TOLERANCE     = 1.5    # cells — close enough = arrived
BUG2_FOLLOW_MAX    = 200    # max wall-follow steps before give up

# Robot states
IDLE           = "IDLE"
ASSIGNED       = "ASSIGNED"
NAVIGATING     = "NAVIGATING"
SCANNING       = "SCANNING"
ZONE_COMPLETE  = "ZONE_COMPLETE"

COLORS = [
    (255, 80,  80),   # robot 0 — red
    (80,  200, 80),   # robot 1 — green
    (80,  150, 255),  # robot 2 — blue
]


class Robot:
    def __init__(self, robot_id, spawn_cx, spawn_cy, arena,
                 odom_sigma=0.02, heading_sigma=0.5,
                 tof_noise_sigma=0.3):
        self.id      = robot_id
        self.arena   = arena
        self.color   = COLORS[robot_id % len(COLORS)]

        # Ground-truth pose (cells)
        self.true_x  = float(spawn_cx)
        self.true_y  = float(spawn_cy)
        self.true_h  = 0.0   # heading degrees, 0=East

        # Estimated pose (dead reckoning + ArUco correction)
        self.est_x   = float(spawn_cx)
        self.est_y   = float(spawn_cy)
        self.est_h   = 0.0

        # Noise params
        self.odom_sigma    = odom_sigma      # cells per step
        self.heading_sigma = heading_sigma   # degrees per step
        self.tof_sigma     = tof_noise_sigma # cells

        # Local log-odds grid
        self.log_odds  = np.full((ARENA_H, ARENA_W), L_PRIOR, dtype=np.float32)
        self.tof_hits  = np.zeros((ARENA_H, ARENA_W), dtype=np.int32)

        # State machine
        self.state         = IDLE
        self.assigned_zone = None   # (x1,y1,x2,y2) cell bbox
        self.goal_cx       = None
        self.goal_cy       = None
        self.scan_step     = 0

        # Bug2
        self.bug2_mode         = "straight"  # "straight" | "wall_follow"
        self.bug2_follow_count = 0
        self.bug2_mline_start  = None  # (x,y) start for m-line
        self.path_history      = [(spawn_cx, spawn_cy)]

        # Metrics
        self.step_count   = 0
        self.aruco_step   = 0
        self.coverage_pct = 0.0
        self.battery      = 100.0   # simulated, drains 0.05/step

        self.rng = np.random.default_rng(robot_id * 42)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assign_zone(self, zone_bbox, zone_goal):
        """Assign exploration zone. zone_bbox=(x1,y1,x2,y2), zone_goal=(cx,cy)."""
        self.assigned_zone = zone_bbox
        self.goal_cx, self.goal_cy = zone_goal
        self.state = ASSIGNED
        self.bug2_mode = "straight"
        self.bug2_follow_count = 0
        self.bug2_mline_start = (self.est_x, self.est_y)

    def step(self):
        """Advance robot one simulation step."""
        self.step_count  += 1
        self.aruco_step  += 1
        self.battery = max(0.0, self.battery - 0.05)

        if self.state == IDLE or self.state == ZONE_COMPLETE:
            return

        if self.state == ASSIGNED:
            self.state = NAVIGATING

        if self.state == NAVIGATING:
            self._navigate_step()
            self._tof_sweep()
            self._aruco_correct()

        elif self.state == SCANNING:
            self._scan_step()

        self._update_coverage()
        self.path_history.append((self.est_x, self.est_y))
        if len(self.path_history) > 2000:
            self.path_history = self.path_history[-2000:]

    # ------------------------------------------------------------------
    # Navigation — Bug2
    # ------------------------------------------------------------------

    def _navigate_step(self):
        gx, gy = float(self.goal_cx), float(self.goal_cy)
        dist   = math.hypot(gx - self.est_x, gy - self.est_y)

        if dist < GOAL_TOLERANCE:
            self.state    = SCANNING
            self.scan_step = 0
            return

        if self.bug2_mode == "straight":
            self._bug2_straight(gx, gy)
        else:
            self._bug2_wall_follow(gx, gy)

    def _bug2_straight(self, gx, gy):
        dx = gx - self.est_x
        dy = gy - self.est_y
        dist = math.hypot(dx, dy)
        if dist == 0:
            return
        ux, uy = dx / dist, dy / dist

        nx = self.true_x + ux * MOVE_SPEED
        ny = self.true_y + uy * MOVE_SPEED
        ncx, ncy = int(round(nx)), int(round(ny))

        if not self.arena.is_occupied(ncx, ncy):
            self._move_to(nx, ny)
        else:
            # Hit obstacle → switch to wall follow
            self.bug2_mode         = "wall_follow"
            self.bug2_follow_count = 0

    def _bug2_wall_follow(self, gx, gy):
        self.bug2_follow_count += 1
        if self.bug2_follow_count > BUG2_FOLLOW_MAX:
            # Give up on this goal — mark zone complete anyway
            self.state = ZONE_COMPLETE
            return

        # Check if m-line is clear ahead and closer to goal
        # M-line: straight line from start to goal
        if self._on_mline_and_closer(gx, gy):
            self.bug2_mode = "straight"
            return

        # Right-hand rule: try right, then forward, then left, then back
        angles = [self.true_h - 90, self.true_h, self.true_h + 90, self.true_h + 180]
        moved  = False
        for angle in angles:
            rad  = math.radians(angle)
            nx   = self.true_x + math.cos(rad) * MOVE_SPEED
            ny   = self.true_y + math.sin(rad) * MOVE_SPEED
            ncx, ncy = int(round(nx)), int(round(ny))
            if not self.arena.is_occupied(ncx, ncy):
                self.true_h = angle % 360
                self._move_to(nx, ny)
                moved = True
                break
        if not moved:
            # Completely surrounded — mark done
            self.state = ZONE_COMPLETE

    def _on_mline_and_closer(self, gx, gy):
        """Check if robot is near m-line and closer to goal than when it hit the obstacle."""
        if self.bug2_mline_start is None:
            return False
        sx, sy = self.bug2_mline_start
        # Distance from current position to m-line (point-to-line)
        line_len = math.hypot(gx - sx, gy - sy)
        if line_len < 0.001:
            return False
        cross = abs((gx - sx) * (self.est_y - sy) - (gy - sy) * (self.est_x - sx)) / line_len
        if cross > 1.5:
            return False
        # Closer to goal than start of wall follow?
        start_dist = math.hypot(gx - sx, gy - sy)
        curr_dist  = math.hypot(gx - self.est_x, gy - self.est_y)
        return curr_dist < start_dist - 1.0

    def _move_to(self, nx, ny):
        """Move ground truth + add odometry noise to estimate."""
        self.true_x = max(1.0, min(ARENA_W - 2.0, nx))
        self.true_y = max(1.0, min(ARENA_H - 2.0, ny))
        noise_x = self.rng.normal(0, self.odom_sigma)
        noise_y = self.rng.normal(0, self.odom_sigma)
        self.est_x  = self.true_x + noise_x
        self.est_y  = self.true_y + noise_y
        self.est_h  = (self.true_h + self.rng.normal(0, self.heading_sigma)) % 360

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def _scan_step(self):
        """Full 360° ToF sweep at current position."""
        for angle_deg in range(0, 360, int(360 / TOF_ANGLES)):
            self._cast_ray(self.est_x, self.est_y, angle_deg)
        self.scan_step += 1
        if self.scan_step >= SCAN_TURNS:
            self.state = ZONE_COMPLETE

    # ------------------------------------------------------------------
    # ToF Sensor
    # ------------------------------------------------------------------

    def _tof_sweep(self):
        """Cast rays in TOF_ANGLES directions from current estimated pose."""
        for i in range(TOF_ANGLES):
            angle_deg = (self.est_h + i * 360 / TOF_ANGLES) % 360
            self._cast_ray(self.est_x, self.est_y, angle_deg)

    def _cast_ray(self, ox, oy, angle_deg):
        """
        DDA ray cast. Updates log_odds grid.
        Cells along ray → L_FREE update.
        First occupied cell → L_OCC update.
        """
        rad = math.radians(angle_deg)
        dx  = math.cos(rad)
        dy  = math.sin(rad)

        for step in range(1, TOF_RANGE_CELLS + 1):
            rx = ox + dx * step
            ry = oy + dy * step
            cx = int(round(rx))
            cy = int(round(ry))

            if cx < 0 or cy < 0 or cx >= ARENA_W or cy >= ARENA_H:
                break

            if self.arena.is_occupied(cx, cy):
                # Add ToF noise — might detect slightly past obstacle
                noisy_range = step + self.rng.normal(0, self.tof_sigma)
                if noisy_range >= step - 0.5:  # still plausible hit
                    self.log_odds[cy, cx] = np.clip(
                        self.log_odds[cy, cx] + L_OCC, L_MIN, L_MAX
                    )
                    self.tof_hits[cy, cx] += 1
                break
            else:
                self.log_odds[cy, cx] = np.clip(
                    self.log_odds[cy, cx] + L_FREE, L_MIN, L_MAX
                )

    # ------------------------------------------------------------------
    # ArUco correction
    # ------------------------------------------------------------------

    def _aruco_correct(self):
        if self.aruco_step < ARUCO_INTERVAL:
            return
        self.aruco_step = 0
        # Convert true pose to metres, get noisy ArUco observation
        true_xm, true_ym = self.true_x * CELL_SIZE, self.true_y * CELL_SIZE
        obs_xm, obs_ym   = self.arena.aruco_observe(true_xm, true_ym)
        # Weighted average: 0.7 ArUco, 0.3 dead reckoning
        obs_cx = obs_xm / CELL_SIZE
        obs_cy = obs_ym / CELL_SIZE
        self.est_x = 0.7 * obs_cx + 0.3 * self.est_x
        self.est_y = 0.7 * obs_cy + 0.3 * self.est_y

    # ------------------------------------------------------------------
    # Coverage metric
    # ------------------------------------------------------------------

    def _update_coverage(self):
        zone = self.assigned_zone
        if zone is None:
            return
        x1, y1, x2, y2 = zone
        zone_cells = (x2 - x1) * (y2 - y1)
        if zone_cells == 0:
            return
        observed = np.sum(self.log_odds[y1:y2, x1:x2] != L_PRIOR)
        self.coverage_pct = min(100.0, observed / zone_cells * 100.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_binary_map(self, threshold=0.5):
        """
        Convert log-odds to binary occupancy for merging.
        Returns: int8 array, 1=occupied, 0=free, -1=unknown
        """
        prob = 1.0 / (1.0 + np.exp(-self.log_odds))
        result = np.full((ARENA_H, ARENA_W), UNKNOWN, dtype=np.int8)
        result[prob > 0.6]  = OCCUPIED
        result[prob < 0.4]  = FREE
        return result

    def get_certainty_map(self):
        """Certainty per cell = |log_odds| / L_MAX, in [0,1]."""
        return np.abs(self.log_odds) / L_MAX
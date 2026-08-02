"""
auction.py — Contract Net Protocol auctioneer.

Zone partitioning:
  Divide arena into N_ROBOTS × N_ROBOTS grid of zones (minus border).
  Each zone is a cell-bounding-box.

Bidding:
  bid(robot, zone) = w1 × dist_to_zone_centroid + w2 × (1 - robot.coverage_pct/100)
  Lower bid = better (closer + less explored wins).

Tie-breaking: lower robot_id wins.
"""

import math
from arena import ARENA_W, ARENA_H
from robot import IDLE, ZONE_COMPLETE

W1 = 0.7   # distance weight
W2 = 0.3   # exploration weight (prefer robots that explored less)


def partition_arena(n_robots, margin=2):
    """
    Partition arena into a pool of zone bboxes.
    Returns list of (x1, y1, x2, y2) cell bboxes.
    """
    # Create n_robots × n_robots zones (more zones than robots = re-auction needed)
    cols = n_robots + 1
    rows = n_robots + 1

    usable_w = ARENA_W - 2 * margin
    usable_h = ARENA_H - 2 * margin
    zone_w   = usable_w // cols
    zone_h   = usable_h // rows

    zones = []
    for r in range(rows):
        for c in range(cols):
            x1 = margin + c * zone_w
            y1 = margin + r * zone_h
            x2 = x1 + zone_w
            y2 = y1 + zone_h
            # Clamp to arena
            x2 = min(x2, ARENA_W - margin)
            y2 = min(y2, ARENA_H - margin)
            if x2 > x1 and y2 > y1:
                zones.append((x1, y1, x2, y2))
    return zones


def zone_centroid(zone):
    x1, y1, x2, y2 = zone
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def compute_bid(robot, zone):
    cx, cy = zone_centroid(zone)
    dist   = math.hypot(cx - robot.est_x, cy - robot.est_y)
    unexplored_ratio = 1.0 - robot.coverage_pct / 100.0
    return W1 * dist + W2 * unexplored_ratio


class Auctioneer:
    def __init__(self, robots, n_robots):
        self.robots         = robots
        self.zone_pool      = partition_arena(n_robots)
        self.assigned_zones = {}   # robot_id -> zone bbox
        self.completed      = set()  # zone indices completed
        self.active_auctions= {}   # robot_id -> zone_index

    # ------------------------------------------------------------------

    def tick(self):
        """
        Called every simulation step.
        Find idle robots, run auction for unassigned zones.
        Returns list of (robot_id, zone_bbox, zone_goal) assignments made this tick.
        """
        idle_robots = [r for r in self.robots
                       if r.state in (IDLE, ZONE_COMPLETE)]
        available_zones = [i for i, z in enumerate(self.zone_pool)
                           if i not in self.completed
                           and i not in self.active_auctions.values()]

        assignments = []
        for robot in idle_robots:
            if not available_zones:
                break
            # Contract Net: robot bids on all available zones
            best_zone_idx = None
            best_bid      = float("inf")
            for zi in available_zones:
                zone = self.zone_pool[zi]
                bid  = compute_bid(robot, zone)
                if bid < best_bid:
                    best_bid      = bid
                    best_zone_idx = zi
                elif bid == best_bid:
                    # Tie: lower robot_id wins
                    if robot.id < self.robots[self.active_auctions.get(
                            best_zone_idx, robot.id)].id:
                        best_zone_idx = zi

            if best_zone_idx is not None:
                zone  = self.zone_pool[best_zone_idx]
                goal  = (int(zone_centroid(zone)[0]),
                         int(zone_centroid(zone)[1]))
                robot.assign_zone(zone, goal)
                self.active_auctions[robot.id] = best_zone_idx
                self.assigned_zones[robot.id]  = zone
                available_zones.remove(best_zone_idx)
                assignments.append((robot.id, zone, goal))

        # Mark completed zones
        for robot in self.robots:
            if robot.state == ZONE_COMPLETE:
                zi = self.active_auctions.get(robot.id)
                if zi is not None and zi not in self.completed:
                    self.completed.add(zi)
                    # Don't pop active_auctions yet — let next tick clear idle

        return assignments

    def all_zones_done(self):
        return len(self.completed) >= len(self.zone_pool)

    def status(self):
        """Return dict of zone pool stats."""
        return {
            "total_zones":     len(self.zone_pool),
            "completed_zones": len(self.completed),
            "active_zones":    len(self.active_auctions),
        }
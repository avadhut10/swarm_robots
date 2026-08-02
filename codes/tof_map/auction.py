"""
Contract Net Protocol auctioneer for zone assignment
"""

import numpy as np

class Auctioneer:
    def __init__(self, grid_size, n_robots):
        self.grid_size = grid_size
        self.n_robots = n_robots
        self.zones = []
        self.w1 = 0.6  # Weight for distance
        self.w2 = 0.4  # Weight for coverage
        
    def set_zones(self, zones):
        """Set the list of zones to be auctioned"""
        self.zones = zones
        
    def run_auction(self, robots, step_count):
        """Run Contract Net Protocol to assign zones to robots"""
        # Get available zones (not assigned and not complete)
        available_zones = [z for z in self.zones if not z['complete'] and z['assigned'] is None]
        
        # Get idle robots
        idle_robots = [r for r in robots if r.state == 'IDLE']
        
        if not available_zones or not idle_robots:
            return
            
        # Each idle robot bids on all available zones
        bids = []
        for robot in idle_robots:
            for zone in available_zones:
                bid_value = self._calculate_bid(robot, zone)
                bids.append({
                    'robot_id': robot.robot_id,
                    'zone': zone,
                    'bid': bid_value
                })
                
        if not bids:
            return
            
        # Sort bids (lowest bid wins)
        bids.sort(key=lambda x: (x['bid'], x['robot_id']))
        
        # Assign zones (avoid double assignment)
        assigned_zones = set()
        assigned_robots = set()
        
        for bid in bids:
            if (id(bid['zone']) not in assigned_zones and 
                bid['robot_id'] not in assigned_robots):
                
                # Assign zone to robot
                zone = bid['zone']
                robot = next(r for r in robots if r.robot_id == bid['robot_id'])
                
                zone['assigned'] = robot.robot_id
                robot.assigned_zone = zone
                robot.state = 'ASSIGNED'
                
                assigned_zones.add(id(zone))
                assigned_robots.add(robot.robot_id)
                
    def _calculate_bid(self, robot, zone):
        """Calculate bid value for robot-zone pair"""
        # Distance component
        dist = np.sqrt(
            (robot.position[0] - zone['center'][0])**2 + 
            (robot.position[1] - zone['center'][1])**2
        )
        max_dist = self.grid_size * 0.1  # Max possible distance
        dist_normalized = dist / max_dist
        
        # Coverage component (prefer less explored robots)
        coverage = robot.get_coverage()
        
        # Combined bid (lower is better)
        bid = self.w1 * dist_normalized + self.w2 * (1 - coverage)
        
        return bid
        
    def get_unassigned_zones(self):
        """Get list of unassigned and incomplete zones"""
        return [z for z in self.zones if not z['complete'] and z['assigned'] is None]
        
    def get_zone_completion(self):
        """Calculate overall zone completion percentage"""
        if not self.zones:
            return 0
        completed = sum(1 for z in self.zones if z['complete'])
        return completed / len(self.zones)